# 
<p align="center">
  <h1 align="center"> <ins>FormerStereo</ins> :<br> Learning Representations from Foundation Models for Domain Generalized Stereo Matching <br> ⭐ECCV 2024⭐</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=odh1uA4AAAAJ">Yongjian Zhang</a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=gbBAujsAAAAJ">Longguang Wang</a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=_kzDdx8AAAAJ">Kunhong Li</a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=8fma3awAAAAJ">Yun Wang</a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=WQRNvdsAAAAJ">Yulan Guo</a>
  </p>
  <h2 align="center"><p>
    <a href="https://link.springer.com/chapter/10.1007/978-3-031-72946-1_9" align="center">Paper</a> | 
    <a href="https://zhangyj85.github.io/FormerStereo.github.io/" align="center">Project Page</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://github.com/zhangyj85/FormerStereo.github.io/blob/main/static/images/pipeline.png?raw=true" alt="example" width=80%>
    <br>
    <em>FormerStereo is a general framework designed to enhance the zero-shot capability of any learning-based stereo matching algorithm.</em>
</p>

## Environment
In your python environment (tested on Ubuntu 22.04, python 3.11, CUDA 12.1), run:
```bash
conda create -n FormerStereo python=3.11
conda activate FormerStereo
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm matplotlib scikit-image
pip install opencv-python
pip install tensorboardX
pip install timm==0.5.4
```
We provide the environment configuration file `environment.yaml` to check the package versions.
Note that not all packages are required for testing.

## How to use
1. Modify the `path` in `create_links.sh`, then create symbolic links for the datasets and model weights:
```bash
sh create_links.sh
```
2. Modify the `root_path` in `./Option/options.json` and create the rootpath JSON file as:
```bash
cd ./data_generator
python main_generator.py
cd ..
```
3. Modify the `mode`, `split`, `data.datasets`, `model.name` and `train.resume` to test the specific pretrained model on the target split of the selected dataset as:
```bash
sh test.sh
```
Note that the default `options.json` is configured to test Former-PSMNet (DINOv2-L) on the `trainingH` split of the Middlebury dataset.
The supported pretrained models include `Former_PSMNet`, `Former_GwcNet`, `Former_CFNet` and `Former_RAFT`.
To switch the Vision Foundation Models (_e.g._, from DINOv2 to SAM), modify the `backbone` argument of the feature extractor.

## SceneFlow Pretrained Models
[Google Drive](https://drive.google.com/drive/folders/1VzunPcrQu2cemc_4Jr2bRL6Cs4Z2KNwv?usp=sharing)

## Reproduce Results
| Model                  | KITTI 2015 (all-Bad 3.0) | KITTI 2012 (all-Bad 3.0) | Middlebury-H (noc-Bad 2.0) | Middlebury-Q (noc-Bad 2.0) | ETH3D (noc-Bad 1.0) |
|------------------------|:----------:|:----------:|:------------:|:------------:|:-----:|
| FormerPSMNet-DAM-L     |    5.00    |    4.22    |     6.95     |     5.73     | 7.74  |
| FormerPSMNet-DINOv2-L  |    4.95    |    3.75    |     7.71     |     6.18     | 6.73  |
| FormerPSMNet-SAM-L     |    5.03    |    4.25    |     9.51     |     7.75     | 6.36  |
| FormerGwcNet-DAM-L     |    5.11    |    3.94    |     6.60     |     4.90     | 4.03  |
| FormerGwcNet-DINOIv2-L |    5.11    |    3.93    |     7.11     |     4.86     | 5.07  |
| FormerCFNet-DAM-L      |    5.09    |    3.89    |     8.40     |     6.00     | 4.40  |
| FormerCFNet-DINOv2-B   |    4.99    |    3.84    |     8.69     |     6.02     | 4.51  |
| FormerRAFT-DAM-L       |    5.18    |    3.94    |     7.97     |     5.60     | 3.51  |

The reproduce results of `Former-PSMNet-DAM-L` and `Former-RAFT-DAM-L` are slightly worse than the results reported in our paper. 
We will investigate these issues in the future.

## Running Time
We measured the average running time of `DAM-L`-integrated models on the KITTI 2015 dataset using an RTX 4090.

| Model     | FormerPSMNet | FormerGwcNet | FormerCFNet | FormerRAFT (32 iters) |
|-----------|:------------:|:------------:|:-----------:|:---------------------:|
| Time (ms) |    384.98    |    407.80    |   419.52    |        496.44         |

## License
All our code except DINOv2 is MIT license.
DINOv2 has an Apache 2 license [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).


## BibTeX
If you find our models useful, please consider citing our paper!
```
@InProceedings{formerstereo_zhangyj_eccv2024,
author="Zhang, Yongjian and Wang, Longguang and Li, Kunhong and Wang, Yun and Guo, Yulan",
title="Learning Representations from Foundation Models for Domain Generalized Stereo Matching",
booktitle="ECCV",
year="2024",
}
```
