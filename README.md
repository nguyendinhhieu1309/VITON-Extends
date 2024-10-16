# Parser-Free Virtual Try-on via Distilling Appearance Flows, CVPR 2021

Official code for CVPR 2021 paper 'Parser-Free Virtual Try-on via Distilling Appearance Flows'

![Python](https://img.shields.io/badge/Python-3.6-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-orange?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green?style=for-the-badge&logo=nvidia)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-red?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![VITON](https://img.shields.io/badge/Dataset-VITON-blueviolet?style=for-the-badge)

---

The training code for our paper is released, and we provide detailed instructions to run it. Below are the required setup details and instructions for testing and training on the VITON dataset.

## Environment Setup (Using `pip`)

To run this project, please install the following dependencies using `pip`:

### Required Libraries

- **Python Version**: Python 3.12
- **Deep Learning Libraries**:
  - PyTorch 2.2.1
  - TorchVision 0.17.1
- **Graphics and Processing**:
  - CUDA 11.8 (Ensure your system has an appropriate GPU and CUDA version installed)
  - CuPy 13.3.0
- **Computer Vision**:
  - OpenCV 4.10.0

### Installation

1. Create and activate a virtual environment using `venv`:

   ```bash
   python3 -m venv tryon_env
   source tryon_env/bin/activate   # On Linux/Mac
   # or
   tryon_env\Scripts\activate      # On Windows
2. Install the necessary libraries using pip:
  ```bash
  pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118
  pip install cupy==13.3.0
  pip install opencv-python==4.10.0
3. Clone the repository:
  ```bash
  git clone https://github.com/nguyendinhhieu1309/VITON-Extends.git
  cd VITON-Extends
