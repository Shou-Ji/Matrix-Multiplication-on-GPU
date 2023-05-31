import torch
import numpy as np
from timeit import default_timer as timer

def Multiply(a, b):
    c = a @ b
    return c
def MultiplyNumpy(a, b):
    c = np.dot(a, b)
    return c
def Warmup(Device):
    cuda_zeros = torch.zeros(1).to(Device)
    non_cuda_zeros = np.zeros(1)
    Multiply(cuda_zeros, cuda_zeros.T)
    Multiply(non_cuda_zeros, non_cuda_zeros.T)
    cuda_zeros.to('cpu')

def main():
    # Initialization
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Device, "\n")

    NumberOfElements = 64000000
    A_cuda = (torch.ones(NumberOfElements, dtype = torch.float32) * 2).to(Device)
    B_cuda = (torch.ones(NumberOfElements, dtype = torch.float32) * 3).to(Device)
    A_non_cuda = np.ones(NumberOfElements, np.float32) * 2
    B_non_cuda = np.ones(NumberOfElements, np.float32) * 3

    Warmup(Device)


    # Measurement
    Start = timer()
    ResultCuda = Multiply(A_cuda, B_cuda.T)
    ExecutionTimeOfMultiply = timer() - Start
    
    Start = timer()
    ResultNumpy = Multiply(A_non_cuda, B_non_cuda.T)
    ExecutionTimeOfMultiplyNumpy = timer() - Start


    # Result
    print(f"{'':<27}{'With GPU':<13}Without GPU")
    print(f"{'Execution Time (second):':<27}{str(ExecutionTimeOfMultiply.__round__(7)):<13}{ExecutionTimeOfMultiplyNumpy.__round__(7)}")
    #print(f"{'Multiply = ':<18} {ResultCuda}")
    #print(f"{'MultiplyNumpy = ':<18} {ResultNumpy}\n")


    # Release
    ResultCuda = ResultCuda.to('cpu')
    A_cuda = A_cuda.to('cpu')
    B_cuda = B_cuda.to('cpu')


if __name__ == '__main__':
    main()

# Output :
#   cuda 
#
#                              With GPU     Without GPU
#   Execution Time (second):   5.6e-05      0.0499461