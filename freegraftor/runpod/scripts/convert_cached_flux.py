"""Lossless key/layout conversion of the already cached FLUX checkpoint.

No new weights, quantization, or numeric cast. Validate native model schemas and
round-trip every tensor through the official Diffusers conversion before saving.
"""
from pathlib import Path
import hashlib, json, os, re, sys
import torch
from safetensors.torch import load_file, save_file

ROOT = Path('/workspace/freegraftor')
SOURCE = Path('/workspace/instantx-0807/cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21')
DEST = ROOT / 'cache/flux-native'
sys.path.insert(0, str(ROOT/'src_snapshot/src'))
os.environ['FLUX_DEV'] = str(DEST)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    torch.set_num_threads(8)
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'conversion.json').exists():
        print('Already converted and verified',flush=True);return
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers, convert_ldm_vae_checkpoint
    from flux.model import Flux
    from flux.modules.autoencoder import AutoEncoder
    from flux.util import configs
    sources = sorted((SOURCE/'transformer').glob('*.safetensors'))
    d={}
    for p in sources:
        print('Loading',p.name,flush=True);d.update(load_file(str(p)))
    n={}
    def move(dst,src): n[dst]=d[src]
    for native,other in [('time_in','timestep_embedder'),('vector_in','text_embedder'),('guidance_in','guidance_embedder')]:
        for a,b in [('in_layer','linear_1'),('out_layer','linear_2')]:
            for x in ['weight','bias']:move(f'{native}.{a}.{x}',f'time_text_embed.{other}.{b}.{x}')
    for a,b in [('txt_in','context_embedder'),('img_in','x_embedder'),('final_layer.linear','proj_out')]:
        for x in ['weight','bias']:move(a+'.'+x,b+'.'+x)
    for i in range(19):
        a=f'double_blocks.{i}';b=f'transformer_blocks.{i}'
        for branch,norm,attn,ff,proj in [('img','norm1','to_','ff','to_out.0'),('txt','norm1_context','add_','ff_context','to_add_out')]:
            for x in ['weight','bias']:
                move(f'{a}.{branch}_mod.lin.{x}',f'{b}.{norm}.linear.{x}')
                keys=[f'{b}.attn.{attn}{q}{"_proj" if branch=="txt" else ""}.{x}' for q in ['q','k','v']]
                n[f'{a}.{branch}_attn.qkv.{x}']=torch.cat([d[k] for k in keys],dim=0)
                move(f'{a}.{branch}_attn.proj.{x}',f'{b}.attn.{proj}.{x}')
                move(f'{a}.{branch}_mlp.0.{x}',f'{b}.{ff}.net.0.proj.{x}')
                move(f'{a}.{branch}_mlp.2.{x}',f'{b}.{ff}.net.2.{x}')
            for q,k in [('query','q'),('key','k')]:
                move(f'{a}.{branch}_attn.norm.{q}_norm.scale',f'{b}.attn.norm_{"added_" if branch=="txt" else ""}{k}.weight')
    for i in range(38):
        a=f'single_blocks.{i}';b=f'single_transformer_blocks.{i}'
        for x in ['weight','bias']:
            move(f'{a}.modulation.lin.{x}',f'{b}.norm.linear.{x}')
            n[f'{a}.linear1.{x}']=torch.cat([d[f'{b}.attn.to_{q}.{x}'] for q in ['q','k','v']]+[d[f'{b}.proj_mlp.{x}']],dim=0)
            move(f'{a}.linear2.{x}',f'{b}.proj_out.{x}')
        for q,k in [('query','q'),('key','k')]:move(f'{a}.norm.{q}_norm.scale',f'{b}.attn.norm_{k}.weight')
    for x in ['weight','bias']:
        s,t=d[f'norm_out.linear.{x}'].chunk(2,dim=0)
        n[f'final_layer.adaLN_modulation.1.{x}']=torch.cat([t,s],dim=0)
    with torch.device('meta'): schema=Flux(configs['flux-dev'].params).state_dict()
    assert set(schema)==set(n), (set(schema)-set(n),set(n)-set(schema))
    assert all(schema[k].shape==n[k].shape for k in n)
    print('Checking transformer round-trip',len(n),flush=True)
    back=convert_flux_transformer_checkpoint_to_diffusers(dict(n))
    assert set(back)==set(d)
    for k in d:
        assert d[k].dtype==back[k].dtype and torch.equal(d[k],back[k]),k
    del back
    print('Saving native transformer',flush=True)
    tmp=DEST/'flux1-dev.safetensors.tmp';save_file({k:v.contiguous() for k,v in n.items()},str(tmp));tmp.replace(DEST/'flux1-dev.safetensors')
    transformer_tensors=len(n)
    del d,n
    d=load_file(str(SOURCE/'vae/diffusion_pytorch_model.safetensors'));n={}
    for k,v in d.items():
        nk=k
        nk=nk.replace('.conv_norm_out.','.norm_out.')
        nk=re.sub(r'encoder\.down_blocks\.(\d+)\.resnets\.(\d+)',r'encoder.down.\1.block.\2',nk)
        nk=re.sub(r'encoder\.down_blocks\.(\d+)\.downsamplers\.0',r'encoder.down.\1.downsample',nk)
        nk=re.sub(r'decoder\.up_blocks\.(\d+)\.resnets\.(\d+)',lambda m:f'decoder.up.{3-int(m[1])}.block.{m[2]}',nk)
        nk=re.sub(r'decoder\.up_blocks\.(\d+)\.upsamplers\.0',lambda m:f'decoder.up.{3-int(m[1])}.upsample',nk)
        nk=re.sub(r'\.mid_block\.resnets\.(\d+)',lambda m:f'.mid.block_{int(m[1])+1}',nk)
        if '.mid_block.attentions.0.' in nk:
            nk=nk.replace('.mid_block.attentions.0.','.mid.attn_1.')
            nk=nk.replace('.group_norm.','.norm.').replace('.to_q.','.q.').replace('.to_k.','.k.').replace('.to_v.','.v.').replace('.to_out.0.','.proj_out.')
            if nk.endswith('.weight') and v.ndim==2:v=v[:,:,None,None].contiguous()
        nk=nk.replace('.conv_shortcut.','.nin_shortcut.')
        assert nk not in n,nk
        n[nk]=v
    with torch.device('meta'):schema=AutoEncoder(configs['flux-dev'].ae_params).state_dict()
    assert set(schema)==set(n),(set(schema)-set(n),set(n)-set(schema))
    assert all(schema[k].shape==n[k].shape for k in n)
    back=convert_ldm_vae_checkpoint(dict(n),json.loads((SOURCE/'vae/config.json').read_text()))
    assert set(back)==set(d),(set(d)-set(back),set(back)-set(d))
    assert all(torch.equal(d[k],back[k]) for k in d)
    tmp=DEST/'ae.safetensors.tmp';save_file(n,str(tmp));tmp.replace(DEST/'ae.safetensors')
    for name in ['text_encoder','text_encoder_2','tokenizer','tokenizer_2']:
        p=DEST/name
        if not p.exists():p.symlink_to(SOURCE/name,target_is_directory=True)
    record=dict(source_snapshot=str(SOURCE),source_blobs={str(p.relative_to(SOURCE)):p.resolve().name for p in sources+[SOURCE/'vae/diffusion_pytorch_model.safetensors']},transformer_tensors=transformer_tensors,vae_tensors=len(n),roundtrip_all_tensors_exact=True,quantization=False,dtype_conversion=False)
    record['native_sha256']={name:sha(DEST/name) for name in ['flux1-dev.safetensors','ae.safetensors']}
    (DEST/'conversion.json').write_text(json.dumps(record,indent=2))
    print('Conversion and exact tensor roundtrip verified',flush=True)

if __name__=='__main__':main()
