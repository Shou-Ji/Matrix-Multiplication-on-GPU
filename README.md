# CUDA-GPU-Parallel-Computing
Measuring the performance of matrix multiplication both with- and without a GPU.

# Environment

Python : 3.8
GPU : NVIDIA GeForce GTX 1070
CUDA Driver : Release 12.1, V12.1.66
CUDA : 11.1
PyTorch : 1.8.0+cu111
NumPy : 1.24.2

# 1. Installation

`git clone [https://github.com/SongHaoHuang/CUDA-GPU-Parallel-Computing.git](https://github.com/SongHaoHuang/CUDA-GPU-Parallel-Computing.git)`

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

This section does not need to be performed for [installation](https://www.notion.so/CUDA-GPU-Parallel-Computing-8e328c93a75a41bab300f19efbc3ebaa).

## 2.1. Windows

Perform the [prepare virtualenv](https://www.notion.so/CUDA-GPU-Parallel-Computing-8e328c93a75a41bab300f19efbc3ebaa).

### 2.1.1. PyTorch + CUDA GPU

Install PyTorch GPU with python wheel package.

```bash
pip install torch==1.8.0+cu111 -f [https://download.pytorch.org/whl/cu111/torch_stable.html](https://download.pytorch.org/whl/cu111/torch_stable.html)
```

- Validation
    
    ```python
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Output : cuda
    ```