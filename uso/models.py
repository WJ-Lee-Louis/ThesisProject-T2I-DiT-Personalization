"""Download only pinned files; never download weights to the local PC."""
from pathlib import Path
from freecus.run import write_json

FILES = {
    'flux': ['flux1-dev.safetensors', 'ae.safetensors'],
    't5': ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model', 'pytorch_model.bin'],
    'clip': ['config.json', 'model.safetensors', 'merges.txt', 'vocab.json', 'tokenizer.json',
             'tokenizer_config.json', 'special_tokens_map.json'],
    'lora': ['uso_flux_v1.0/dit_lora.safetensors', 'uso_flux_v1.0/projector.safetensors'],
    'siglip': ['config.json', 'preprocessor_config.json', 'model.safetensors'],
}


def stage(lock, destination):
    from huggingface_hub import snapshot_download
    paths = {}
    if set(lock) != set(FILES):
        raise ValueError('Unexpected model lock')
    for key, model in lock.items():
        print('Staging', model['repo'], model['revision'], flush=True)
        paths[key] = snapshot_download(model['repo'], revision=model['revision'], allow_patterns=FILES[key])
        if any(not (Path(paths[key]) / name).is_file() for name in FILES[key]):
            raise ValueError('Incomplete model snapshot: ' + key)
    write_json(destination, paths)
    return paths
