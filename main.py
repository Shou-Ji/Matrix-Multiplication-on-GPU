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
    print(f"{'MultiplyCuda = ':<18} {ResultCuda}")
    print(f"{'MultiplyNumpy = ':<18} {ResultNumpy}\n")

    print(f"{'':<20}{'With GPU':<15}Without GPU")
    print(f"{'RunTime (second):':<20}{str(RunTimeOfMultiplyCuda.__round__(7)):<15}{RunTimeOfMultiplyNumpy.__round__(7)}")


    # Release
    ResultCuda = ResultCuda.to('cpu')


if __name__ == '__main__':
    main()

# Output :
#   cuda 
#
#   MultiplyCuda =     tensor([6., 6., 6.,  ..., 6., 6., 6.], device='cuda:0')
#   MultiplyNumpy =    [6. 6. 6. ... 6. 6. 6.]
#
#                       With GPU       Without GPU
#   RunTime (second):   0.0031947      0.1176511