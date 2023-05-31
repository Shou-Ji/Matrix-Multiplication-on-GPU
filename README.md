# CUDA-GPU-Parallel-Computing
Measuring the execution time of matrix multiplication with and without a GPU.

![matrix multiplication](https://github.com/SongHaoHuang/Matrix-Multiplication-on-GPU/assets/128410674/b6095217-4432-461c-8201-c1ac2f0e5a19)

![output](https://github.com/SongHaoHuang/Matrix-Multiplication-on-GPU/assets/128410674/1f14e600-2833-42b6-9ad5-13a4b393990a)

Note that a `Warmup` prior to measurements is performed to exclude one-time start-up costs.

# Environment
GPU : NVIDIA GeForce GTX 1070 8G

CPU : Intel(R) Xeon(R) CPU E3-1230 v3 @ 3.30GHz


Python : 3.8

PyTorch : 1.8.0+cu111 (CUDA : 11.1)

NumPy : 1.24.2

CUDA Driver : Release 12.1, V12.1.66

# 1. Installation

`git clone https://github.com/SongHaoHuang/CUDA-GPU-Parallel-Computing.git`

## 1.1. Windows

### 1.1.1. Prepare Virtualenv

Install virtualenv.

```bash
pip install virtualenv
```

Create a virtual environment and activate it.

```bash
# Find your python directory
where python

# Create a virtual environment
cd \My\Project\Directory
virtualenv --python C:\Path\To\Python\python.exe venv

# Activate the virtual environment just created
.\venv\Scripts\activate
```

### 1.1.2. Prepare Requirements

Install the required libraries.

```bash
# Navigate to project directory
cd \My\Project\Directory

# Install libraries
pip install -r requirements.txt
```

## 1.2. Linux

### 1.2.1. Prepare Virtualenv

Install virtualenv.

```bash
pip3 install virtualenv
```

Create a virtual environment and activate it

```bash
# Navigate to project directory
cd \My\Project\Directory

# Create a virtual environment
# venv is the name of your virtual environment
virtualenv -p /usr/bin/python3 venv

# Activate the virtual environment just created
source venv/bin/activate
```

### 1.2.2. Prepare Requirements

Install the required libraries.

```bash
pip install -r requirements.txt
```

# 2. Library Dependency

This section does not need to be performed for installation.

## 2.1. Windows

Perform the **1.1.1. prepare virtualenv** section.

### 2.1.1. PyTorch + CUDA GPU

Install PyTorch GPU with python wheel package.

```bash
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

- Validation
    
    ```python
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Output : cuda
    ```
