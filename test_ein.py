import torch

if __name__ == '__main__':
    print('Test 1: ii->i')
    t1 = torch.arange(0, 9).reshape(3, 3)
    print(torch.einsum('ii->i', t1))
    print('======================')

    print('Test 2: ij->ji')
    t1 = torch.arange(0, 6).view(2, 3)
    print(torch.einsum('ij->ji', t1))
    print('======================')

    print('Test 3: ij->')
    t1 = torch.arange(6).reshape(2, 3)
    print(torch.einsum('ij->', t1))
    print('======================')

    print('Test 4: ij->j')
    t1 = torch.arange(6).reshape(2, 3)
    print(torch.einsum('ij->j', t1))
    print('======================')

    print('Test 5: ik,k->i')
    t1 = torch.arange(6).reshape(2, 3)
    t2 = torch.arange(3)
    print(torch.einsum('ik,k->i', t1, t2))
    print('======================')

    print('Test 6: ik,kj->ij')
    t1 = torch.arange(6).reshape(2, 3)
    t2 = torch.arange(15).reshape(3, 5)
    print(torch.einsum('ik,kj->ij', t1, t2))
    print('======================')

    print('Test 7: i,i->')
    t1 = torch.arange(6)
    print(torch.einsum('i,i->', t1, t1))
    print('======================')

    print('Test 8: ij,ij->')
    t1 = torch.arange(6).reshape(2, 3)
    t2 = torch.arange(6, 12).reshape(2, 3)
    print(torch.einsum('ij,ij->', t1, t2))
    print('======================')

    print('Test 9: i,j->ij')
    t1 = torch.arange(1, 6)
    t2 = torch.arange(6, 11)
    print(torch.einsum('i,j->ij', t1, t2))
    print('======================')

    print('Test 10: ijk,ikl->ijl')
    t1 = torch.arange(30).reshape(2, 3, 5)
    t2 = torch.arange(40).reshape(2, 5, 4)
    print(torch.einsum('ijk,ikl->ijl', t1, t2))
    print('======================')

    print('Test 11: pqrs,tuqvr->pstuv')
    t1 = torch.arange(120).reshape(2, 3, 4, 5)
    t2 = torch.arange(288).reshape(2, 4, 3, 3, 4)
    print(torch.einsum('pqrs,tuqvr->pstuv', t1, t2))
    print('======================')

    print('Test 12: ik,jkl,il->ij')
    t1 = torch.arange(6).reshape(2, 3)
    t2 = torch.arange(105).reshape(5, 3, 7)
    t3 = torch.arange(14).reshape(2, 7)
    print(torch.einsum('ik,jkl,il->ij', t1, t2, t3))
    print('======================')

    print('Test 13: ijk->ikj')
    t1 = torch.arange(24).reshape(2, 3, 4)
    print(torch.einsum('ijk->ikj', t1))
