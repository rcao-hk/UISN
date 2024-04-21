# Uncertainty-aware Suction Grasping for Cluttered Scenes [RA-Letter 2024]

Official repository for paper "Uncertainty-aware Suction Grasping for Cluttered Scenes" 

![Image Title](https://github.com/rcao-hk/UISN/blob/master/framework.png)

# Dataset

Download data and labels from our [SuctionNet webpage](https://graspnet.net/suction).

# Environment

The code has been tested with `CUDA 11.6` and `pytorch 1.13.0` on ubuntu `20.04`

# Installation

Create new enviornment:
```bash
conda create --name grasp python=3.8
```
Activate the enviornment and install Pytorch 1.13.0:
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install Minkowski Engine:
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
Install prerequisites:
```bash
pip install -r requirements.txt
```
Install suctionnetAPI:
```bash
git clone https://github.com/intrepidChw/suctionnms.git
cd suctionnms
pip install .

git clone https://github.com/graspnet/suctionnetAPI
cd suctionnetAPI
pip install .
```
# Data Preparation
1. Precompute normal map for scenes:
```bash
cd dataset
python generate_normal_data.py --dataset_root '/path/to/SuctionNet/dataset'
```
2. Download suction label and dense point clouds from https://graspnet.net/suction, and extract files into dataset root.
3. Precompute suction label for scenes:
```bash
cd dataset
python generate_suction_data.py --dataset_root '/path/to/SuctionNet/dataset'
```
4. Download segmentation mask from [UOIS](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187973_link_cuhk_edu_hk/EcLzwCXsPUdNiix_mxqwVmcBODekt_Qfj6DSWPgHzqXUGA?e=iC1ouY), [UOAIS](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187973_link_cuhk_edu_hk/EcYqOfP2P5NIn9AGgNbWj1IBSqnDRz-cIfajjJrYMXrDXw?e=SUXnSE).

# Training

For training, use the following command:

```
bash scripts/train.sh
```

For evaluation, use the following command, where 'xxxx' denotes splits:'seen', 'similar' or 'novel': 

```
bash scripts/eval_xxxx.sh
```

# Pre-trained Models
Comming soon.

## Citation

if you find our work useful, please cite

```
@ARTICLE{USIN_grasp,
  author={Cao, Rui and Yang, Biqi and Li, Yichuan and Fu, Chi-Wing and Heng, Pheng-Ann and Liu, Yun-Hui},
  journal={IEEE Robotics and Automation Letters}, 
  title={Uncertainty-Aware Suction Grasping for Cluttered Scenes}, 
  year={2024},
  volume={9},
  number={6},
  pages={4934-4941},
  keywords={Grasping;Uncertainty;Point cloud compression;Robots;Noise measurement;Three-dimensional displays;Predictive models;Deep learning in grasping and manipulation;perception for grasping and manipulation;computer vision for automation},
  doi={10.1109/LRA.2024.3385609}}

```

