# PoseGaussian: Pose-Driven Novel View Synthesis for Human Representation

<a ng-if="options.download" ng-href="/api/repo/PoseGaussian/zip" target="__self" class="btn btn-outline-primary btn-sm ng-scope" href="/api/repo/PoseGaussian/zip">Download Repository</a>




This repository is the official implementation of PoseGaussian: Pose-Driven Novel View Synthesis for Human Representation.

[![Watch the demo](https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/thumbnail.png)](https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4)

## Requirements
linux=22.04.5,
python=3.10.13,
cuda=11.8
## Dataset
Train on THuman2.0

Download render_data and real_data from URL below extract the data in PoseGaussian folder

## Installation
Configure the environment
```setup
conda env create --file environment.yml

conda activate PoseGaussian
```
Install diff-gaussian-rasterization
```setup
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
conda activate PoseGaussian
pip install -e submodules/diff-gaussian-rasterization
```
 RAFT-Stereo implementation of the correlation sample
 
```setup
cd ..
git clone https://github.com/princeton-vl/RAFT-Stereo.git
python setup.py install
cd ../..

```
Install Pose detector MMPose

#More detailed Installation steps in https://mmpose.readthedocs.io/en/latest/installation.html

```setup
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
```
```setup
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
mim install "mmdet>=3.1.0"
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
```
```setup
cp "PATH TO ROOT FOLDER/heatmap.py" .
```
```setup
conda deactivate
cd ..
```
# Training
```setup
python trainer.py
```
# Testing
```setup
python test.py \
--test_data_root 'real_data' \
--ckpt_path 'PATH/TO/PoseGaussian_pose.pth' \
--src_view 0 1 \
--ratio=0.5
```

