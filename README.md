
# 👗 Enhancing Pose Adaptability in Virtual Try-On Systems

**Nguyen Dinh Hieu, Tran Minh Khuong, Phan Duy Hung[0000-0002-6033-6484]**  
FPT University, Hanoi, Vietnam  
[hieundhe180318@fpt.edu.vn](mailto:hieundhe180318@fpt.edu.vn), [khuongtmhe180089@fpt.edu.vn](mailto:khuongtmhe180089@fpt.edu.vn), [hungpd2@fe.edu.vn](mailto:hungpd2@fe.edu.vn)

---

## ✨ Abstract

Garment fitting in virtual try-on systems often struggles with complex body poses, occlusions, and misalignments. This project introduces a novel solution that improves adaptability and garment warping using a global appearance flow estimation model. Our method leverages a StyleGAN-based architecture, incorporating a global style vector to enhance spatial alignment between garments and body poses. We also integrate a flow refinement module for finer garment deformation. The results, tested on the VITON benchmark, highlight the effectiveness of our approach, especially under challenging conditions, achieving state-of-the-art performance.

---

## 📚 Table of Contents

- [📄 Paper](#paper)
- [🛠️ Environment Setup](#environment-setup)
- [💻 Installation](#installation)
- [🚀 Training](#training)
- [🔍 Testing](#testing)
- [📊 Dataset](#dataset)
- [🤝 Acknowledgements](#acknowledgements)

---

## 📄 Paper

- [Official Paper](https://doi.org/10.1007/978-981-96-4606-7_21)

---

## 🛠️ Framework and Environment Setup

This project utilizes the following frameworks and libraries:

| **Framework** | **Version** |
|:--------------|:------------|
| [![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red?logo=pytorch&logoColor=white)](https://pytorch.org) | 2.2.1+cu118 |
| [![TorchVision](https://img.shields.io/badge/TorchVision-0.17.1-yellow?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html) | 0.17.1+cu118 |
| [![CuPy](https://img.shields.io/badge/CuPy-13.3.0-blue?logo=cupy&logoColor=white)](https://cupy.dev) | 13.3.0 | 
| [![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green?logo=opencv&logoColor=white)](https://opencv.org) | 4.10.0 | 
| [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org) | 3.12 |
| [![TensorboardX](https://img.shields.io/badge/TensorboardX-2.4-orange?logo=tensorflow&logoColor=white)](https://github.com/lanpa/tensorboardX) | 2.4 |

---

## 💻 Installation

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
   git clone https://github.com/nguyendinhhieu1309/VITON-Extends.git
   cd VITON-Extends
   ```

---

## 🚀 Training on VITON-Extends Dataset

1. **Download VITON-Extends Training Set**  
   Download the VITON training set from [VITON-Extends_train](https://drive.google.com/drive/folders/1wsIp7n2msLdNLffNo4EEKPfWZZK_284w?usp=drive_link) and place the folder `VITON-Extends_traindata` under the folder `dataset`.

2. **Download VGG-19 Model**  
   Download the VGG_19 model from [VGG_Model](https://drive.google.com/drive/folders/1LmrMmyXWSUlne7ES25kvQrolUIW8YTTQ?usp=drive_link) and place `vgg19-dcbb9e9d.pth` under the folder `models`.

3. **Train the Parser-Based Network**  
   Run the following scripts to train the parser-based warping module:  
   ```bash
   scripts/train_VITON-Extends_stage1.sh
   scripts/train_VITON-Extends_e2e.sh
   ```

4. **Train the Parser-Free Network**  
   After training the parser-based network, run:  
   ```bash
   scripts/train_VITON-Extends2_stage1.sh
   scripts/train_VITON-Extends2_e2e.sh
   ```

---

## 🔍 Testing

1. **Download the Checkpoints**  
   Download the checkpoints from [here](https://drive.google.com/drive/folders/15AbTw16w13dN1hY430flBZbe1EfumJT7?usp=drive_link) and place the folder `VITON-Extends` under the folder `checkpoints`.  
   The folder `checkpoints/VITON-Extends` should contain `warp_model_final.pth` and `gen_model_final.pth`.

2. **Run the Test**  
   Run the following command to test the saved model:  
   ```bash
   python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0
   ```

3. **Test FID**
   
   Download Dataset All: To download various datasets for training and testing the model, use the following links [All models](https://drive.google.com/drive/folders/10r1cMHbfpEF3jCH5JqHM1o-9B-Q4KNRB?usp=drive_link)
   
   For evaluating the Fréchet Inception Distance (FID) score, follow the instructions [FID](https://github.com/mseitzer/pytorch-fid.git) to set up the FID calculation. This will allow you to compare the quality of generated images with real ones.
---

## 📊 Dataset

VITON-Extends contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each featuring front-view woman images and top clothing items at a resolution of 1024×768. These provide a diverse range of body poses for improved garment fitting. To train from scratch, download the [VITON-Extends_train](https://drive.google.com/drive/folders/1wsIp7n2msLdNLffNo4EEKPfWZZK_284w?usp=drive_link). For testing, download the [VITON-Extends_test](https://drive.google.com/drive/folders/1wsIp7n2msLdNLffNo4EEKPfWZZK_284w?usp=drive_link).
## 🎨 Results Demo
![VITON-Extends Results](https://github.com/user-attachments/assets/df7df550-f3c6-431c-b2d6-384011173b60)

**Results of VITON-Extends Model Applied to Virtual Try-On in Complex Poses**

The qualitative results, as illustrated in Figure, further support the quantitative findings. The proposed model demonstrates superior performance in generating realistic try-on images, particularly in challenging poses and occlusion scenarios. When the subject is standing sideways and has crossed arms, the model consistently preserves garment alignment and visual integrity, outperforming other models that struggle with these challenges.

---

## 🤝 Acknowledgements 

This project builds upon the principles of prior research in virtual try-on systems. The base code was adapted from "Clothflow: A flow-based model for clothed person generation." All contributions are acknowledged, and further citations are provided within the source code and accompanying documentation.
