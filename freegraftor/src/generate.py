import os

_missing_model_paths = [
    name for name in ("FLUX_DEV", "GROUNDING_DINO", "SAM")
    if not os.environ.get(name)
]
if _missing_model_paths:
    raise RuntimeError(
        "Set the required model path environment variables before importing "
        f"FreeGraftor: {', '.join(_missing_model_paths)}"
    )
# os.environ["KORNIA_LAZY_LOADER_INSTALL_MODE"] = "off"

# from kornia.utils.config import set_lazyload_installation_mode
# set_lazyload_installation_mode("off")

import kornia
from kornia.config import kornia_config
kornia_config.lazyloader.installation_mode = 'auto'

import argparse
from pathlib import Path
import json 
from collections import defaultdict
import torch
from pipeline import FreeGraftorPipeline, ConceptConfig, GenerationConfig

import time

def main(args, models=None):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    pipeline = FreeGraftorPipeline(
        models=models, 
        image_cache_dir=args.image_cache_dir,
        image_info_cache_dir=args.image_info_cache_dir,
        requires_offload=args.requires_offload
    )
    
    concept_configs = []
    config_path = Path(args.concept_config_path)
    
    if config_path.is_file():
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_concept_configs = json.load(f)
        concept_configs.extend(ConceptConfig(**config) for config in raw_concept_configs)
    else:
        concept_configs.extend(
            ConceptConfig(class_name=class_name, image_path=image_path) 
            for class_name, image_path in zip(args.class_names, args.image_paths)
        )
            
    seed = args.start_seed
    
    width = args.width if args.width % 16 == 0 else args.width - args.width % 16
    height = args.height if args.height % 16 == 0 else args.height - args.height % 16
    inject_block_ids = list(range(args.start_inject_block, args.end_inject_block + 1))
    
    for idx in range(args.num_images):   
        info = {
            'seed': seed,
            'num_steps': args.num_steps,
            'guidance': args.guidance,
            'width': width,
            'height': height,
            'start_inject_step': args.start_inject_step,
            'end_inject_step': args.end_inject_step,
            'inject_block_ids': inject_block_ids,
            'sim_threshold': args.sim_threshold,
            'cyc_threshold': args.cyc_threshold,
            'inject_match_dropout': args.inject_match_dropout
        }
        
        t0 = time.time()
        
        gen_image = pipeline(
            concept_configs=concept_configs,
            prompt=args.gen_prompt,
            template_prompt=args.template_prompt,
            template_path=args.template_path,
            output_dir=args.output_dir,
            clear_image_cache=args.clear_image_cache,
            clear_image_info_cache=args.clear_image_info_cache,
            config=GenerationConfig(**info),
        )
        
        t1 = time.time()
        print(f"\n\nTime elapsed: {t1 - t0:.2f} seconds")
        
        m = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Max memory allocated: {m:.2f} MB")
        
        seed += 1
    
    return models

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_prompt', type=str, default="")
    parser.add_argument('--gen_prompt', type=str, default="A can is placed on a desk, next to a laptop.")
    parser.add_argument('--template_path', type=str, default='')
    parser.add_argument('--concept_config_path', type=str, default='configs/can.json')  
    parser.add_argument('--class_names', nargs='+', default=[]) 
    parser.add_argument('--image_paths', nargs='+', default=[])
    
    parser.add_argument('--image_cache_dir', type=str, default='./image_cache')
    parser.add_argument('--image_info_cache_dir', type=str, default='./image_info_cache')
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--clear_image_cache', action='store_true')
    parser.add_argument('--clear_image_info_cache', action='store_true')
    
    parser.add_argument('--guidance', type=float, default=3)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument(
        '--requires_offload',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable CPU offload (use --no-requires-offload to disable it).',
    )
    parser.add_argument('--modules_to_offload', nargs='+', default=['ae', 'collage_creator'])
    parser.add_argument('--num_images', type=int, default=2)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    
    parser.add_argument('--start_inject_step', type=int, default=0)
    parser.add_argument('--end_inject_step', type=int, default=25)
    parser.add_argument('--start_inject_block', type=int, default=0)
    parser.add_argument('--end_inject_block', type=int, default=56)
    parser.add_argument('--sim_threshold', type=float, default=0.2)
    parser.add_argument('--cyc_threshold', type=float, default=1.5)
    parser.add_argument('--inject_match_dropout', type=float, default=0.4)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)
