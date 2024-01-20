import torch

if __name__ == '__main__':

    print('Test 1 ======================')
    t1 = torch.arange(0, 60).view(3,4,5)
    print(t1)
    t2 = torch.mean(t1,1,dtype=torch.float32)
    print(t2)
    