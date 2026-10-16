import os
import argparse

import json
from pathlib import Path
import omegaconf
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
from scripts.utils import postprocess
from scripts.utils import load_data_for_worker_dcm432x250x250
from scripts.utils import reverse_2_original_dimension_432x250x250
from scripts.utils import load_data_for_worker_dcm96x192x288
from conds import cond_fn_factary
from conds import cond_indeces_factary

def main(
        name,
        save_dir='',

        num_samples=9,
        batch_size=1,
        base_samples="",
        base_file_type="dcm",
        cond_samples="",
        cond_file_type="npy",
        slice_start=90,
        slice_end=186,

        model_path="",
        cond_fn_type = "",
        cond_kwargs = {},
        cond_indeces_type = "default",
        conda_indeces_kwargs = {},

        large_size=96,
        small_size=96,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        resblock_updown=False,
        rescale_learned_sigmas=False,
        learn_sigma=True,
        class_cond=False,
        use_scale_shift_norm=True,
        use_checkpoint=False,
        use_fp16=False,

        acc_timesteps=-1,
        diffusion_steps=1000,
        rescale_timesteps=False,
        noise_schedule="linear",
        clip_denoised=True,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
):
    dist_util.setup_dist()
    logger.configure(dir=save_dir)
    os.makedirs(save_dir, exist_ok=True)

    logger.log("creating model...")          
    model, diffusion = sr_create_model_and_diffusion(
        large_size=large_size,
        small_size=small_size,
        class_cond=class_cond,
        learn_sigma=learn_sigma,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        timestep_respacing=timestep_respacing,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if use_fp16:
        model.convert_to_fp16()
    model.eval()

    cond_indeces = cond_indeces_factary(cond_indeces_type, conda_indeces_kwargs)
    if cond_fn_type != "":
        cond_fn_ = cond_fn_factary(cond_fn_type)
        try:
            cond_kwargs = omegaconf.OmegaConf.to_container(
                cond_kwargs, resolve=True, throw_on_missing=False
            )
        except ValueError:
            cond_kwargs = {}
    else:
        cond_fn_ = None
        cond_kwargs = {}

    logger.log("loading data...")
    
    data = load_data_for_worker_dcm96x192x288(base_samples, slice_start, slice_end, "dcm")
    if cond_samples != "":
        cond = load_data_for_worker_dcm96x192x288(cond_samples, slice_start, slice_end, "npy")
    else:
        cond = None

    logger.log("creating samples...")
    all_images = []

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    data = data.to(device)
    if cond is not None:
        cond = cond.to(device)

    total_needed = num_samples
    total_iter = (total_needed + batch_size - 1) // batch_size  # 向上取整计算总步数
    pbar = tqdm(total=total_needed, desc="Generating Samples", unit="sample", dynamic_ncols=True)
    current_count = 0  # 记录已生成样本数

    model_kwargs = { 'low_res': data }

    if cond is not None:
        cond_kwargs_ = { 'low_res': cond }
        cond_kwargs.update(cond_kwargs_)
    else:
        cond_kwargs = {}
    print(model_kwargs['low_res'].shape)
    shape = (batch_size, 1, model_kwargs['low_res'].shape[2], model_kwargs['low_res'].shape[3], model_kwargs['low_res'].shape[4])
    # A fixed random seed was used to maintain consistent initial noise across different patches from the same dataset
    if device == "cuda":
        th.cuda.manual_seed_all(10)
    else:
        th.manual_seed(10)
    noise = th.randn(*shape, device=device)
    print(f"acc timesteps: {acc_timesteps}") 

    sample = diffusion.p_sample_loop(
        model,
        shape, 
        noise,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        progress=True,
        cond_fn=cond_fn_,
        cond_kwargs=cond_kwargs,
        dps_indeces=cond_indeces,
        acc_timesteps=acc_timesteps,
    )
    
    sample = sample.permute(0, 1, 3, 4, 2)
    sample = sample.contiguous() 

    out_path = os.path.join(logger.get_dir(), "sample.npy")
    np.save(out_path, sample.cpu().numpy())
    
    logger.log(f"created {len(all_images) * batch_size} samples\n")

    pbar.close()

    final_result = sample.cpu().numpy()
    final_result = postprocess(final_result)

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in final_result.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{datetime.datetime.now().strftime('%H%M%S%f')}.npy")
        logger.log(f"saving to {out_path}")
        np.save(out_path, final_result)

    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = omegaconf.OmegaConf.load(args.config)

    main(name=name, **config)
