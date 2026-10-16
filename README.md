# GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction

This repository is the official code release for the paper **GR-Diffusion**, which proposes a framework that combines a three-dimensional discrete Gaussian Representation (GR) with a diffusion model for low-dose whole-body PET reconstruction. The overall pipeline is: **first, the GR module (`DGR/`) infers a geometric-prior condition from projection data; then, a 3D diffusion model (`DDPM_3D/`) generates the final reconstruction under the guidance of that condition**. These two core stages correspond to the two folders in the repository.

---

## Table of Contents

- [Paper Information](#paper-information)
- [Method Overview](#method-overview)
- [Summary of Experimental Results](#summary-of-experimental-results)
- [Repository Structure](#repository-structure)
- [Core Pipeline and the Two Modules](#core-pipeline-and-the-two-modules)
  - [Stage 1: DGR (Gaussian Representation Condition Inference)](#stage-1-dgr-gaussian-representation-condition-inference)
  - [Stage 2: DDPM_3D (Gradient-Guided Diffusion Generation)](#stage-2-ddpm_3d-gradient-guided-diffusion-generation)
- [Other Folders](#other-folders)
- [Environment and Dependencies](#environment-and-dependencies)
- [Quick Start](#quick-start)
- [Citation](#citation)

---

## Paper Information

- **Title**: GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction
- **Authors**: Mengxiao Geng, Zijie Chen, Ran Hong, Bingxuan Li, Qiegen Liu
- **Affiliations**: School of Information Engineering, Nanchang University (M. Geng, Z. Chen, R. Hong, Q. Liu); Anhui Province Key Laboratory of Biomedical Imaging and Intelligent Processing, Institute of Artificial Intelligence, Hefei Comprehensive National Science Center (B. Li)
- **Corresponding authors**: Bingxuan Li, Qiegen Liu
- **Co-first authors**: Mengxiao Geng, Zijie Chen

### Abstract

Positron emission tomography (PET) reconstruction is a critical challenge in molecular imaging, often hampered by noise amplification, structural blurring, and detail loss due to sparse sampling and the ill-posed nature of inverse problems. In this work, we propose **GR-Diffusion**, a framework that synergistically integrates the geometric priors of a discrete Gaussian Representation (GR) with the generative power of diffusion models for 3D low-dose whole-body PET reconstruction.

GR-Diffusion first employs GR to generate a reference 3D PET image from projection data, establishing a physically grounded and structurally explicit benchmark that overcomes the low-pass limitations of conventional point-based or voxel-based methods. This reference image then serves as a **dual guide** during the diffusion process, ensuring both global consistency and local accuracy. Specifically, we design a **hierarchical guidance mechanism** based on the GR reference: fine-grained guidance leverages differences to refine local details, while coarse-grained guidance uses multi-scale difference maps to correct deviations. This allows the diffusion model to sequentially integrate the strong geometric prior from GR and recover sub-voxel information.

Experimental results on the UDPET and Clinical datasets with varying dose levels show that GR-Diffusion outperforms state-of-the-art methods in enhancing 3D whole-body PET image quality and preserving physiological details.

### Main Contributions

1. **A unified GR-Diffusion framework for PET reconstruction**: the first framework that seamlessly integrates an explicit 3D GR with a diffusion model for 3D PET reconstruction, establishing a new paradigm that couples explicit geometry with generative refinement.
2. **A theory-grounded gradient guidance mechanism**: a novel gradient guidance strategy rooted in mathematical analysis, which proves that the GR-derived reference image can surrogate ground-truth gradients under uncertainty, enabling anatomically consistent guidance during diffusion sampling.
3. **Multi-scale hierarchical refinement**: a dual-path guidance with fine-grained and coarse-grained components, allowing sequential integration of geometric priors for detail recovery and noise suppression.

### Keywords

Low-dose PET reconstruction, sinogram, 3D Gaussian representation, diffusion model, gradient-guided.

---

## Method Overview

GR-Diffusion consists of two core modules (see Fig. 2 / Fig. 3 in the paper):

1. **3D Gaussian Representation module (GR)**: constructs a physically grounded reference image from raw sinogram data.
2. **Gradient-guided diffusion module**: takes the GR output as a geometric prior to steer the reverse diffusion process for detail recovery and noise suppression.

**Stage 1 (GR)** models the 3D activity image $\lambda$ as a linear combination of discrete, parameterized, isotropic 3D Gaussian primitives (center $\mu_i$, standard deviation $\sigma_i$, intensity $I_i$). Reconstruction proceeds in three steps — "discretize & align to the voxel grid", "fast volume reconstruction", and "global optimization in the sinogram domain (Poisson likelihood + TV regularization)" — together with an adaptive density control strategy based on pruning, cloning, and splitting.

**Stage 2 (Gradient-guided diffusion)** pre-trains a 3D DDPM with a 3D-UNet backbone. The GR-reconstructed image $x_{GR}$ serves as a geometric prior, and gradient guidance terms steer the reverse sampling:

- **Fine-grained guidance**: in the early phase, when the discrepancy between $x_t$ and the ground truth $x_{gt}$ is large, $\nabla_{x_t}\|x_t - x_{GR}\|_1$ effectively substitutes for $\nabla_{x_t}\|x_t - x_{gt}\|_1$ (see Theorem 1 in the paper).
- **Coarse-grained guidance**: in the later phase, multi-scale Gaussian kernels ($3\times3\times3$, $\sigma = 1.0, 2.0, 4.0$) smooth the predicted and reference images before computing gradients, suppressing spurious fine-scale signals from $x_{GR}$.

Both are activated only during the middle interval $[0.4T, 0.6T]$ of the reverse process, balancing computational cost and guidance strength.

---

## Summary of Experimental Results

On the UDPET (MICCAI 2024 Ultra-low Dose PET Imaging Challenge) and Clinical (19 patients, DigitMI 930 PET/CT) datasets, GR-Diffusion is compared with UNet-2D, UNet-3D, DDPM-2D, DDPM-3D, R²-Gaussian, X²-Gaussian, and DGR using PSNR↑ / SSIM↑ / MSE↓ / FID↓:

| Dataset | DRF | Best PSNR (GR-Diffusion) | Gain over the second-best method |
| --- | --- | --- | --- |
| UDPET | 20 | **44.37 dB** | +0.48 dB (second-best DDPM-3D) |
| UDPET | 50 | **43.53 dB** | +1.05 dB (second-best DDPM-3D) |
| Clinical | 4 | **32.10 dB** | optimal |
| Clinical | 10 | **31.31 dB** | optimal |

Ablation studies verify that the dual-path guidance combining fine-grained and coarse-grained terms (Type Ⅳ) outperforms configurations using a single guidance term or no guidance across all three views (coronal, sagittal, axial).

---

## Repository Structure

```
GRDPET/
├── DGR/                 # Stage 1: 3D Gaussian Representation; infers the geometric-prior condition
├── DDPM_3D/             # Stage 2: gradient-guided 3D diffusion model; generates the final PET image
├── Dataset/             # Dataset preprocessing (DICOM → npy / SUV conversion, background filtering, etc.)
├── Metrics_eval/        # Evaluation scripts (PSNR/SSIM/MSE/FID, LPIPS, charts and notebooks)
├── UNet/                # Baseline: UNet-2D
├── UNet_3D/             # Baseline: UNet-3D
├── ViT/                 # Baseline: ViT
├── IDDPM_2D/            # Baseline: DDPM-2D (improved diffusion)
├── MPRNet/              # Baseline: MPRNet (image restoration network)
├── r2_gaussian/         # Baseline: R²-Gaussian
├── x2-gaussian/         # Baseline: X²-Gaussian
├── annotator.html       # npy image annotation tool
├── .gitignore
└── README.md
```

---

## Core Pipeline and the Two Modules

### Stage 1: DGR (Gaussian Representation Condition Inference)

`DGR/` implements the 3D discrete Gaussian representation reconstruction described in Section II-A of the paper, transforming low-dose sinograms (or projection data) into a structurally explicit reference image $x_{GR}$.

- `DGR/DGR.py`: the `DGR` class — isotropic Gaussian representation, fast volume reconstruction, and adaptive density control (pruning / cloning / splitting).
- `DGR/cone_beam_svct.py`: reconstruction entry point for the whole-body 360×360 (673 slices) resolution.
- `DGR/cone_beam_svct_384x250x250.py` / `cone_beam_svct_432x250x250.py`: reconstruction entry points for the 384 / 432 slices, 250×250 resolutions.
- `DGR/config/*.sh`: run scripts for different datasets / dose levels (e.g., `reconstruction_D20_p1.sh`, `reconstruction_D4_LK.sh`).
- `DGR/CT/`: dataset readers.
- `DGR/params/`: model / optimization / pipeline parameters (adapted from the Inria 3DGS codebase).
- `DGR/utils/`: utilities for fast volume reconstruction, loss functions, projection operators, etc.
- `DGR/stack_part_2_wholebody.py`: stitches sub-volumes back into the whole-body volume.

Example output: `DGR/output/D20_p1/recon_673x360x360.npy` (this file is the condition input for Stage 2).

### Stage 2: DDPM_3D (Gradient-Guided Diffusion Generation)

`DDPM_3D/` implements the gradient-guided diffusion process described in Section II-B of the paper, using the GR reference image as a geometric prior to guide a pre-trained 3D DDPM toward the final reconstruction.

- `DDPM_3D/scripts/train.py`: trains the 3D DDPM (3D-UNet backbone).
- `DDPM_3D/test_dcm_96.py` / `test_dcm_432.py` / `test_dcm_673.py`: inference entry points that take a YAML config via `--config`.
- `DDPM_3D/config/*.yaml`: inference configs, including whole-body (D4/D10/D20/D50), fine/coarse-grained brain, and ablation configs; `cond_samples` points to the DGR output and `model_path` points to the pre-trained weights.
- `DDPM_3D/cond_fn/`: base class for the guidance function.
- `DDPM_3D/conds/`: concrete guidance terms (`low_res_mse`, `low_res_mae`, `gaussian_mse`, `gaussian_and_mae`) and timestep-index factories (`single_range`, `range_intervals`, `equal_intervals`, `random_sample`, `full_range`).
- `DDPM_3D/guided_diffusion/`: diffusion code adapted from improved-diffusion (3D-UNet, sampling loops with `cond_fn`, etc.).
- `DDPM_3D/dcm_to_npy.py` / `proc.py`: DICOM → npy / SUV conversion.
- `DDPM_3D/apply_mask.py`: applies a mask to the reconstruction result.
- `DDPM_3D/model_space/model.pt`: pre-trained 3D DDPM weights.

Mapping between guidance terms and code (see Eq. 16 / Eq. 17 in the paper):

| Guidance type | `cond_fn_type` | Description |
| --- | --- | --- |
| Fine-grained | `low_res_mae` | L1-distance gradient, $\nabla_{x_t}\|x_t - x_{GR}\|_1$ |
| Coarse-grained | `gaussian_mse` | MSE gradient after multi-scale Gaussian smoothing |
| Dual-path (ours) | `gaussian_and_mae` | combination of fine-grained and coarse-grained |

---

## Other Folders

- **Dataset/**: data preprocessing scripts that convert DICOM series into npy volumes and perform SUV normalization (`dim2npy*.py`, `filter_background.py`).
- **Metrics_eval/**: quantitative and perceptual evaluation scripts (`psnr_ssim.py`, `mse_ca.py`, `lpips_fid.py`, `main.py`) and plotting notebooks.
- **UNet/, UNet_3D/, ViT/, IDDPM_2D/, MPRNet/, r2_gaussian/, x2-gaussian/**: baseline methods compared in the paper, each with training and inference scripts.
- **annotator.html**: an npy image annotation tool for manual inspection / labeling of reconstruction results.

---

## Environment and Dependencies

This repository contains multiple sub-projects and does not pin versions uniformly. The main dependencies include:

- Python 3, PyTorch (CUDA)
- Numeric / image: NumPy, OpenCV, SciPy, Pillow
- Medical imaging: `pydicom`, `tigre` (TIGRE cone-beam CT toolbox), `odl` (Operator Discretization Library, with ASTRA), `astra-toolbox`
- Diffusion: `blobfile`, `omegaconf`, `tqdm`, `icecream`
- Gaussian representation: `plyfile`
- Evaluation: `lpips`, `pytorch-fid`

> Note: `DGR/` depends on TIGRE and ODL (ASTRA) for forward/back projection; the diffusion code for `DDPM_3D/` is bundled in `guided_diffusion/`. It is recommended to prepare a separate conda environment for each module.

---

## Quick Start

Taking `D20_p1` from the UDPET dataset (dose reduction factor 20, patient 1) as an example, the full pipeline is as follows:

### 1. Data Preparation

Convert the DICOM data into npy volumes with SUV normalization and place them under the corresponding directories in `Dataset/Dicom/` (see `Dataset/dim2npy*.py` and `DDPM_3D/dcm_to_npy.py`).

### 2. Stage 1: Infer the Condition with DGR

```bash
cd DGR
bash config/reconstruction_D20_p1.sh
```

This script runs `cone_beam_svct.py` and outputs the geometric-prior reference image:

```
DGR/output/D20_p1/recon_673x360x360.npy
```

### 3. Stage 2: Generate the Final Result with DDPM_3D

```bash
cd DDPM_3D
python test_dcm_96.py --config config/inference_wholeBody_D20_p1.yaml
```

This config:

- uses `base_samples` (low-dose DICOM) as input;
- uses `cond_samples` (the GR output `../DGR/output/D20_p1/recon_673x360x360.npy` from the previous step) as the guidance condition;
- applies dual-path gradient guidance via `cond_fn_type: gaussian_and_mae`;
- loads `model_space/model.pt` to perform sampling, with results saved to `DDPM_3D/output/D20_p1/`.

For other doses / datasets, simply replace the corresponding `.sh` script and `.yaml` config (e.g., `reconstruction_D50_p1.sh` → `inference_wholeBody_D50_p1.yaml`, `reconstruction_D4_LK.sh` → `inference_wholeBody_D4_LK.yaml`).

---

## Citation

If this work is helpful to your research, please cite:

```bibtex
@article{geng2025grdiffusion,
  title   = {GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction},
  author  = {Geng, Mengxiao and Chen, Zijie and Hong, Ran and Li, Bingxuan and Liu, Qiegen},
  journal = {IEEE Transactions on ... (under review)},
  year    = {2025}
}
```

Open-source code: <https://github.com/yqx7150/GR-Diffusion>
