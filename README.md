# PoseGaussian: Pose-Driven Novel View Synthesis for Human Representation



This repository is the official implementation of PoseGaussian: Pose-Driven Novel View Synthesis for Human Representation.

[![Watch the demo](https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/thumbnail.png)](https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4)

<a ng-if="options.download" ng-href="https://anonymous.4open.science/api/repo/PoseGaussian/zip" target="__self"  href="https://anonymous.4open.science/api/repo/PoseGaussian/zip" >[Source Code]</a>

<a href="https://anonymous.4open.science/r/PoseGaussian/docs/PoseGaussian.pdf" target="_blank">[Paper Draft]</a>

## Customized Dataset Sample

<table>
  <tr>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>


## Requirements
linux=22.04.5,
python=3.10.13,
cuda=11.8
## Dataset
Train on THuman2.0

Download render_data and real_data from URL below extract the data in PoseGaussian folder

[Training: render_data_data](https://udayton0-my.sharepoint.com/:u:/g/personal/dasguptas2_udayton_edu/Eb9h6FKKqf9Cq0Q0ynIGVhcB1FPJ98EAnXipzGRrYK7SdA?e=JaOYpN)

[Testing: real_data](https://udayton0-my.sharepoint.com/:u:/g/personal/dasguptas2_udayton_edu/EQGmdOFq_qpDnNMpJeOuPxIB0OUFgWyyNJRTQb7GP_oRDQ?e=t5imkG)

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
Copy the heatmap.py to mmpose folder
```setup
cp "PATH TO ROOT FOLDER/heatmap.py" .
```
Run heatmap.py
```setup
python heatmap.py
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
Testing data samples

<table>
  <tr>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
    <td width="200">
    <video width="200" controls>
        <source src="https://raw.githubusercontent.com/sohomd/PoseGaussian/assets/Demo.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </td>
  </tr>
</table>

Run the command with changing path input parameters

```setup
python test.py \
--test_data_root 'real_data' \
--ckpt_path 'PATH/TO/PoseGaussian_pose.pth' \
--src_view 0 1 \
--ratio=0.5
```

