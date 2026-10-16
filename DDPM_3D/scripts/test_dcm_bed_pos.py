import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import datetime
from tqdm import tqdm
from icecream import ic

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from scripts.utils import load_data_for_worker_dcm360p, reverse_2_original_dimension_dcm360p, postprocess
from conds import cond_fn_factary
from conds import cond_indeces_factary

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    cond_fn_ = cond_fn_factary(args.cond_fn) if args.cond_fn != "" else None
    conda_indeces_config = json.loads(args.conda_indeces_config) if args.conda_indeces_config is not None else {}
    cond_indeces = cond_indeces_factary(args.cond_indeces_name, conda_indeces_config)
    cond_kwargs = json.loads(args.cond_kwargs) if args.cond_kwargs is not None else {}

    logger.log("loading data...")
    data = load_data_for_worker_dcm360p(args.base_samples, args.batch_size, args.type_file)
    logger.log("creating samples...")
    all_images = []

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    total_needed = args.num_samples
    total_iter = (total_needed + args.batch_size - 1) // args.batch_size  # 向上取整计算总步数
    pbar = tqdm(total=total_needed, desc="Generating Samples", unit="sample", dynamic_ncols=True)
    current_count = 0  # 记录已生成样本数

    bed_pos = args.bed_pos

    for i in range(bed_pos):
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

    ic(model_kwargs["low_res"].shape)

    shape = (args.batch_size, 1, model_kwargs['low_res'].shape[2], model_kwargs['low_res'].shape[3], model_kwargs['low_res'].shape[4])

    # A fixed random seed was used to maintain consistent initial noise across different patches from the same dataset
    if device == "cuda":
        th.cuda.manual_seed_all(10)
    else:
        th.manual_seed(10)
    noise = th.randn(*shape, device=device)

    if args.acc_timesteps < 1000:
        sample = diffusion.ddim_sample_loop(
            model,
            shape, 
            noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            acc_timesteps=args.acc_timesteps,
            cond_fn=cond_fn_,
        )
    else:
        sample = diffusion.p_sample_loop(
            model,
            shape, 
            noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            cond_fn=cond_fn_,
            cond_kwargs=cond_kwargs,
            dps_indeces=cond_indeces,
        )
    
    sample = sample.contiguous() 

    final_result = sample.squeeze(0).squeeze(0).cpu().numpy()
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in final_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{datetime.datetime.now().strftime('%H%M%S%f')}")
        logger.log(f"saving to {out_path}")
        np.save(out_path, final_result)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        save_dir='',
        type_file="dcm",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="",
        bed_pos=0,
        acc_timesteps=1000,
        cond_fn = "",
        cond_indeces_name = "default",
        conda_indeces_config = None,
        cond_kwargs = None,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
