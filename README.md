# GTA: Geometric Transform-Attention Network for Enhanced Spatial Reasoning in Image Captioning

This repository contains the implementation of **GTA (Geometric Transform-Attention Network)** for image captioning.

GTA extends the X-Linear Attention Network (X-LAN) framework by introducing:
- descriptor-space geometric transformations (**translation, rotation, scaling, and shearing**),
- enriched geometric descriptors (**bounding box coordinates, aspect ratio, area, centroid, perimeter, and compactness**),
- a structured attention refinement pipeline with **CAM**, **SAM**, and **RAN**, and
- improved spatial reasoning for complex multi-object image captioning.

## Paper

**Mohammad Alamgir Hossain, Zhongfu Ye, Md. Bipul Hossen, Md Shohidul Islam, and Md. Ibrahim Abdullah**  
**GTA: Geometric Transform-Attention Network for Enhanced Spatial Reasoning in Image Captioning**  
**Journal:** Neurocomputing  
**DOI:** [10.1016/j.neucom.2026.133497](https://doi.org/10.1016/j.neucom.2026.133497)

If you use this repository, please cite:

```bibtex
@article{hossain2026gta,
  title   = {GTA: Geometric Transform-Attention Network for Enhanced Spatial Reasoning in Image Captioning},
  author  = {Hossain, Mohammad Alamgir and Ye, Zhongfu and Hossen, Md. Bipul and Islam, Md Shohidul and Abdullah, Md. Ibrahim},
  journal = {Neurocomputing},
  year    = {2026},
  pages   = {133497},
  doi     = {10.1016/j.neucom.2026.133497}
}
```

If you also use the original X-LAN baseline framework, please cite:

```bibtex
@inproceedings{xlinear2020cvpr,
  title={X-Linear Attention Networks for Image Captioning},
  author={Pan, Yingwei and Yao, Ting and Li, Yehao and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Requirements
- Python 3
- CUDA 10
- numpy
- tqdm
- easydict
- [PyTorch](http://pytorch.org/) (>1.0)
- [torchvision](http://pytorch.org/)
- [coco-caption](https://github.com/ruotianluo/coco-caption)

## Data Preparation
1. Download the [bottom-up features](https://github.com/peteanderson80/bottom-up-attention) and convert them to `.npz` files:
   ```bash
   python2 tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_10_100
   ```

2. Download the MS COCO caption annotations into the `mscoco` folder. More details on data preparation can be found in [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).

3. Download [coco-caption](https://github.com/ruotianluo/coco-caption) and set the path of `__C.INFERENCE.COCO_PATH` in `lib/config.py`.

## Pretrained Models and Test Results

### Cross Entropy Loss:
- **Pretrained Model**: [Download Link](https://drive.google.com/file/d/1lAMEvP49rR2CiF8kPamlY_MeF4OK_8sv/view?usp=drive_link)
- **Test**: [Test Results Link](https://drive.google.com/file/d/1PVyyb5mCaloRw3ayYtwiKQb-Q1Y7cmhQ/view?usp=drive_link)

### CIDEr Optimization Loss:
- **Pretrained Model**: [Download Link](https://drive.google.com/file/d/155UxpQH1cg2lIAUR-r55w0YU6lhee08f/view?usp=drive_link)
- **Test**: [Test Results Link](https://drive.google.com/file/d/1VWmNVno-fp_WpJ9T1hNgdRfHuRqzkwb8/view?usp=drive_link)

## Training

> **Note:** This repository keeps the original X-LAN experiment folder names for backward compatibility.

### Train GTA model
```bash
bash experiments/xlan/train.sh
```

### Train GTA model with self-critical sequence training
Copy the pretrained model into `experiments/xlan_rl/snapshot` and run:
```bash
bash experiments/xlan_rl/train.sh
```

### Train transformer-based variant
```bash
bash experiments/xtransformer/train.sh
```

### Train transformer-based variant with self-critical sequence training
Copy the pretrained model into `experiments/xtransformer_rl/snapshot` and run:
```bash
bash experiments/xtransformer_rl/train.sh
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python3 main_test.py --folder experiments/model_folder --resume model_epoch
```

## Notes
- GTA is built on top of the X-LAN image captioning framework and retains a compatible training and evaluation pipeline.
- The main modifications are the transformed geometric descriptor bank and the GTA attention refinement modules.
- Folder names inherited from the original X-LAN repository are intentionally preserved for compatibility and reproducibility.

## Acknowledgements
This implementation is built upon the original [X-Linear Attention Networks for Image Captioning](https://arxiv.org/pdf/2003.14080.pdf) codebase and also benefits from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and the PyTorch community.
