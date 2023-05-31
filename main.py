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
    cpu_zeros = np.zeros(1)
    Multiply(cuda_zeros, cuda_zeros.T)
    Multiply(cpu_zeros, cpu_zeros.T)
    cuda_zeros.to('cpu')

def main():
    # Initialization
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Device, "\n")

    NumberOfElements = 64000000
    A_cuda = (torch.ones(NumberOfElements, dtype = torch.float32) * 2).to(Device)
    B_cuda = (torch.ones(NumberOfElements, dtype = torch.float32) * 3).to(Device)
    A_cpu = np.ones(NumberOfElements, np.float32) * 2
    B_cpu = np.ones(NumberOfElements, np.float32) * 3

    Warmup(Device)


    # Measurement
    Start = timer()
    ResultCuda = Multiply(A_cuda, B_cuda.T)
    ExecutionTimeOfMultiply = timer() - Start
    
    Start = timer()
    ResultCpu = Multiply(A_cpu, B_cpu.T)
    ExecutionTimeOfMultiplyCpu = timer() - Start


    # Result
    print(f"{'':<27}{'With GPU':<13}Without GPU")
    print(f"{'Execution Time (second):':<27}{str(ExecutionTimeOfMultiply.__round__(7)):<13}{ExecutionTimeOfMultiplyCpu.__round__(7)}")
    #print(f"{'Multiply = ':<18} {ResultCuda}")
    #print(f"{'MultiplyCpu = ':<18} {ResultCpu}\n")


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