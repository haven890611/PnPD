# PnPD (Polyp Nodule? Dual-path Detection + Segmentation)

This repository contains a dual-path framework for polyp analysis, combining:
- A **detection branch** (YOLO-series) for bounding box prediction, and
- A **segmentation branch** (multiple segmentation backbones) for mask prediction,
with notebooks to run end-to-end inference, dataset surveys, and class-agnostic validation.

> Important: This repo includes both **my own code** and **third-party source code**.  
> Please read the **License & Third-party Code** section before redistribution.

---

## Repository Structure

### `det_branch/`
- **Source code of YOLOv13** (third-party).
- Includes `explainer.ipynb` (my notebook) for inspecting model weights and visualizing Grad-CAM results.

### `seg_branch/`
Segmentation path used in the dual-path framework.  
This folder includes multiple segmentation models:
- **UNeXt**
- **UNetV2**
- **EMCAD**
- **Polyp-PVT**

Training is performed using **each model’s official source code** (third-party).  
Inside each model folder, there is a `dual.ipynb` notebook to run full dual-path inference:
- The segmentation model is combined with a selected YOLO model.
- **Requires `ultralytics`** to be installed.

### `test_single.ipynb`
Notebook for **class-agnostic evaluation**:
- Focuses on whether the predicted box overlaps the GT (ignoring subtype),
- i.e., “single-class” evaluation for localization-only performance.

### `dataset.ipynb`
Notebook to summarize **dataset diversity and composition**:
- statistics and distributions of dataset attributes / categories.

---