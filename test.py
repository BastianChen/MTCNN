import torch

# iou = torch.Tensor([0.1])
# sigma = torch.Tensor([0.5])
# s = torch.exp(-(iou)**2/sigma)
# print(s)


# a = torch.Tensor([1, 5, 1, 6]).reshape(2, 2)
# b = torch.eq(a, 1)
# c = torch.masked_select(a, b)
# print(a)
# print(c.shape[0])

# 通道混洗测试
    # a = torch.Tensor(np.arange(72)).reshape(2, 4, 3, 3)
    # print(a.shape)
    # print(a)
    # b = a.permute(0, 1, 3, 2).reshape(2, 4, 3, 3)
    # # b = a.reshape(2, 2, 2, 3, 3).permute(0, 2, 1, 3, 4).contiguous().view(2, 4, 3, 3)
    # print(a.shape)
    # print(b)
    # print(b - a)
    # a = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    # print(a)
    # a = a.reshape(2, 1, 4)
    # print(a)
    # a = a.permute(1, 0, 2)
    # print(a)
    # a = a.reshape(2, 4)
    # print(a)
