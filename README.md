
# Enhancing Pose Adaptability in Virtual Try-On Systems

**Nguyen Dinh Hieu, Tran Minh Khuong, Phan Duy Hung**  
FPT University, Hanoi, Vietnam  
[hieundhe180318@fpt.edu.vn](mailto:hieundhe180318@fpt.edu.vn), [khuongtmhe180089@fpt.edu.vn](mailto:khuongtmhe180089@fpt.edu.vn), [hungpd2@fe.edu.vn](mailto:hungpd2@fe.edu.vn)

---

## Abstract

Accurate garment fitting in virtual try-on systems remains a significant challenge, particularly when addressing complex body poses, occlusions, and extreme misalignments. To overcome these issues, we introduce a novel approach that enhances both pose adaptability and garment warping by leveraging a global appearance flow estimation model. This work utilizes a StyleGAN-based architecture, incorporating a global style vector to capture whole-image context, which improves spatial alignment between garments and body poses. To further enhance fine-grained details, we integrate a flow refinement module that focuses on local garment deformation.

Experimental results on the VITON benchmark showcase the superiority of our method, particularly under challenging conditions, establishing a new state-of-the-art in virtual try-on technology.

---

## Table of Contents

- [Paper](#paper)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)
- [Dataset](#dataset)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Paper

- [Official Paper](https://github.com/khuong16/Pose-Adapt-VITON-Extends/tree/master)
- [Supplementary Material](#)
- [Checkpoints for Test](#)

---

## Framework and Environment Setup

This project is built using the following frameworks and libraries:

| **Framework** | **Version** | **Icon** |
|:--------------|:------------|:--------:|
| [![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red?logo=pytorch&logoColor=white)](https://pytorch.org) | 2.2.1+cu118 | ![PyTorch Icon](https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg) |
| [![TorchVision](https://img.shields.io/badge/TorchVision-0.17.1-yellow?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html) | 0.17.1+cu118 | ![TorchVision Icon](https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg) |
| [![CuPy](https://img.shields.io/badge/CuPy-13.3.0-blue?logo=cupy&logoColor=white)](https://cupy.dev) | 13.3.0 | ![CuPy Icon](https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png) |
| [![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green?logo=opencv&logoColor=white)](https://opencv.org) | 4.10.0 | ![OpenCV Icon](https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg) |
| [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org) | 3.12 | ![Python Icon](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg) |

---

## Installation

1. **Create a Conda Environment**  
   ```bash
   conda create -n tryon python=3.6
   conda activate tryon
   ```

2. **Install PyTorch**  
   ```bash
   conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
   ```

3. **Install Other Dependencies**  
   ```bash
   conda install cupy
   pip install opencv-python
   ```

4. **Clone the Repository**  
   ```bash
   git clone https://github.com/geyuying/PF-AFN.git
   cd PF-AFN
   ```

---

## Training on VITON Dataset

1. **Download VITON Training Set**  
   Download the VITON training set from [VITON_train](#) and place the folder `VITON_traindata` under the folder `dataset`.

2. **Download VGG-19 Model**  
   Download the VGG_19 model from [VGG_Model](#) and place `vgg19-dcbb9e9d.pth` under the folder `models`.

3. **Train the Parser-Based Network**  
   Run the following scripts to train the parser-based warping module:  
   ```bash
   scripts/train_PBAFN_stage1.sh
   scripts/train_PBAFN_e2e.sh
   ```

4. **Train the Parser-Free Network**  
   After training the parser-based network, run:  
   ```bash
   scripts/train_PFAFN_stage1.sh
   scripts/train_PFAFN_e2e.sh
   ```

---

## Testing

1. **Download the Checkpoints**  
   Download the checkpoints from [here](#) and place the folder `PFAFN` under the folder `checkpoints`.  
   The folder `checkpoints/PFAFN` should contain `warp_model_final.pth` and `gen_model_final.pth`.

2. **Run the Test**  
   Run the following command to test the saved model:  
   ```bash
   bash
   ```

---

## Dataset

VITON contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each of which has a front-view woman photo and a top clothing image with the resolution 256 x 192. Our saved model is trained on the VITON training set and tested on the VITON test set. To train from scratch on VITON training set, you can download VITON_train. To test our saved model on the complete VITON test set, you can download VITON_test.

---

## License

The use of this code is RESTRICTED to non-commercial research and educational purposes.

---

## Acknowledgements

Our code is based on the implementation of "Clothflow: A flow-based model for clothed person generation" (See the citation below), including the implementation of the feature pyramid networks (FPN) and the ResUnetGenerator, and the adaptation of the cascaded structure to predict the appearance flows. If you use our code, please also cite their work as below.

---

## Citation

If our code is helpful to your work, please cite:

@article{ge2021parser,
  title={Parser-Free Virtual Try-on via Distilling Appearance Flows},
  author={Ge, Yuying and Song, Yibing and Zhang, Ruimao and Ge, Chongjian and Liu, Wei and Luo, Ping},
  journal={arXiv preprint arXiv:2103.04559},
  year={2021}
}
