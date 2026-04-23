# Align then Refine: Text-Guided 3D Prostate Lesion Segmentation

Our paper, ***Align then Refine: Text-Guided 3D Prostate Lesion Segmentation***, has been accepted for presentation at the 48th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (**IEEE EMBC 2026**).

This README describes how to reproduce the customized training and inference pipeline in this fork (multi-encoder + text conditioning + attention-based refinement).

## 1) What to run

Main scripts used in this repo:

- `tools/train/run_text_single.sh`: base text trainer run
- `tools/train/run_text_attn_best.sh`: attention run with tuned defaults + pretrained checkpoint
- `tools/infer/run_text_test.sh`: inference + optional evaluation
- `tools/eval/compute_segmentation_metrics.py`: extra metric report

Tools are organized by purpose:

- `tools/train`: training and finetuning launchers
- `tools/infer`: prediction/inference launchers
- `tools/eval`: postprocessing and metric utilities

## 2) Environment

```bash
cd <PROJECT_ROOT>

# 1) Create the base nnU-Net v2 environment (follow the official nnU-Net setup)
conda create -n nnunetv2_repro python=3.10 -y
conda activate nnunetv2_repro

# install this fork in editable mode
pip install -e .

# 2) Extra dependency used by this text-guided pipeline
# (install if not already present in your nnU-Net environment)
pip install open-clip-torch
```

## 3) Required paths

Set these before training or inference:

```bash
export nnUNet_raw=<NNUNET_RAW>
export nnUNet_preprocessed=<NNUNET_PREPROCESSED>
export nnUNet_results=<NNUNET_RESULTS>
```

## 4) Dataset and fold defaults

Examples below use:

- Dataset: `Dataset2203_picai_split`
- Config: `3d_fullres`
- Plans: `nnUNetPlans`
- Fold: `1`

Adjust as needed.

## 5) Training and Inference Workflow

### Stage A: Base text model

```bash
# QUICK=0 is closer to full training
QUICK=0 bash tools/train/run_text_single.sh 0 1 Dataset2203_picai_split 3d_fullres nnUNetPlans tversky
```

Args:

- `<GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS] [LOSS]`
- `[LOSS]`: `dice | tversky | focal_tversky` (mapped internally to `*_topk`)

Common env vars:

- `QUICK` (`1` debug/faster, `0` fuller run)
- `NNUNET_TRAINER`, `NNUNET_PRETRAINED_WEIGHTS`
- `NNUNET_RESULTS_TAG` / `NNUNET_RESULTS_DIR`
- `NNUNET_ITERS_PER_EPOCH`, `NNUNET_VAL_ITERS`
- `NNUNET_TEXT_PROMPTS`, `NNUNET_TEXT_MODEL`, `NNUNET_TEXT_EMBED_DIM`
- `NNUNET_TEXT_MODULATION` (`none|film|gate`)
- `NNUNET_USE_ALIGNMENT_HEAD`, `NNUNET_RETURN_HEATMAP`
- `NNUNET_LAMBDA_ALIGN`, `NNUNET_LAMBDA_HEAT`
- `NNUNET_AUX_WARMUP_EPOCHS`, `NNUNET_AUX_RAMP_EPOCHS`

Loss options for `run_text_single.sh` (6th argument):

- `dice`
- `tversky`
- `focal_tversky`

Examples:

```bash
# Dice + TopKCE
QUICK=0 bash tools/train/run_text_single.sh 0 1 Dataset2203_picai_split 3d_fullres nnUNetPlans dice

# Tversky + TopKCE
QUICK=0 bash tools/train/run_text_single.sh 0 1 Dataset2203_picai_split 3d_fullres nnUNetPlans tversky

# Focal-Tversky + TopKCE
QUICK=0 bash tools/train/run_text_single.sh 0 1 Dataset2203_picai_split 3d_fullres nnUNetPlans focal_tversky
```

Notes:

- `tools/train/run_text_attn_best.sh` launches with `tversky`.
- You can still override the internal loss selection via env, e.g. `NNUNET_TEXT_LOSS=dice_topk` before launch.

Expected output model folder:

```text
<NNUNET_RESULTS>/
  Dataset2203_picai_split/
  nnUNetTrainerMultiEncoderUNetText__nnUNetPlans__3d_fullres/fold_1
```

### Stage B: Attention fine-tuning

```bash
# auto-loads fold checkpoint_best.pth from Stage A if present
QUICK=0 bash tools/train/run_text_attn_best.sh 0 1 Dataset2203_picai_split 3d_fullres nnUNetPlans
```

Args:

- `<GPU_ID> [FOLD] [DATASET] [CONFIG] [PLANS] [PRETRAINED_CKPT]`

Key env vars:

- `NNUNET_CROSS_GAMMA_INIT` (default `0.10`)
- `NNUNET_CROSS_ALPHA` (default `0.12`)
- `NNUNET_CROSS_TAU` (default `0.44`)
- `ATTN_WARMUP_EPOCHS` (default `0`)
- `BASE_LR_REFINER` (default `5e-4`)
- `NNUNET_PRETRAINED_WEIGHTS` (override preload checkpoint)
- `NNUNET_RESULTS_TAG` (output experiment tag)

## 6) Inference

```bash
bash tools/infer/run_text_test.sh <INPUT_DIR> <OUTPUT_DIR> 0 <MODEL_PATH>
```

- `INPUT_DIR`: nnUNet-style input images directory
- `OUTPUT_DIR`: prediction output directory
- `0`: GPU id

Args:

- `<INPUT_DIR> <OUTPUT_DIR> [GPU_ID] [MODEL_PATH]`
- `[MODEL_PATH]`: optional model path. This can be a checkpoint file, a `fold_*` directory, or a model/results directory.

Key env vars:

- `NNUNET_DATASET`, `NNUNET_CONFIG`, `NNUNET_PLANS`
- `NNUNET_TEST_TRAINER`, `NNUNET_FOLD`, `NNUNET_CHECKPOINT_FILE`
- `NNUNET_SKIP_EVAL`, `NNUNET_GT_DIR`

## 7) Optional: extra metric computation

```bash
python tools/eval/compute_segmentation_metrics.py \
  --pred-dir <OUTPUT_DIR> \
  --gt-dir <GT_DIR> \
  --output-dir <METRIC_OUT_DIR>
```

## Acknowledgement

This project is built on top of the excellent nnU-Net framework:

- https://github.com/MIC-DKFZ/nnUNet

We gratefully acknowledge and thank the nnU-Net authors and contributors for open-sourcing and maintaining this powerful toolkit.

## Citation

Citation for our paper, *Align then Refine: Text-Guided 3D Prostate Lesion Segmentation*, will be added/updated here once final publication details are available.

If you use this repository in academic work, please also cite the original nnU-Net paper:

Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z

```bibtex
@article{isensee2021nnu,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nat Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z}
}
```
