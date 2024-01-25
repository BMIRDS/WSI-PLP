import torch

def zero_padding(t, n, dim=0, padding=0):
    assert dim <= 1
    size = list(t.size())
    size[dim] = n
    o = torch.zeros(size, dtype=t.dtype, device=t.device)
    if padding != 0:
        o[:] = padding
    if dim == 0:
        o[:t.size(0), ] = t[:n]
    else:
        o[:,:t.size(1), ] = t[:,:n,]
    return o    