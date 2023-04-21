import torch
import numpy as np
from timeit import default_timer as timer

def MultiplyCuda(a, b):
    c = a * b
    return c
def MultiplyNumpy(a, b):
    c = np.multiply(a, b)
    return c

def main():
    # Initialization
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(Device, "\n")

    NumberOfElements = 64000000
    A_np = np.ones(NumberOfElements, np.float32) * 2
    B_np = np.ones(NumberOfElements, np.float32) * 3

    A_cuda = torch.from_numpy(A_np).to(Device)
    B_cuda = torch.from_numpy(B_np).to(Device)


    # Measurement
    Start = timer()
    ResultCuda = MultiplyCuda(A_cuda, B_cuda)
    RunTimeOfMultiplyCuda = timer() - Start
    
    Start = timer()
    ResultNumpy = MultiplyNumpy(A_np, B_np)
    RunTimeOfMultiplyNumpy = timer() - Start


    # Result
    print(f"{'MultiplyCuda = ':<20} {ResultCuda}")
    print(f"{'MultiplyNumpy = ':<20} {ResultNumpy}\n")

    print(f"{'':<10}{'MultiplyCuda':<25}MultiplyNumpy")
    print(f"{'RunTime':<10}{str(RunTimeOfMultiplyCuda):<25}{RunTimeOfMultiplyNumpy}")


    # Release
    ResultCuda = ResultCuda.to('cpu')


main()

# Output :
#   cuda 
#   
#   MultiplyCuda =       tensor([6., 6., 6.,  ..., 6., 6., 6.], device='cuda:0')
#   MultiplyNumpy =      [6. 6. 6. ... 6. 6. 6.]
#   
#             MultiplyCuda             MultiplyNumpy
#   RunTime   0.0029516999999996685    0.15093200000000007