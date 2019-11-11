import torch
from torch import nn
torch.cuda.is_available()

torch.cuda.device_count()
torch.cuda.current_device()
print(torch.cuda.get_device_name(0))
x = torch.tensor([1, 2, 3])
x = x.cuda(0)
print(x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
x = torch.tensor([1, 2, 3], device=device)
x = torch.tensor([1, 2, 3]).to(device)

y = x**2
print(y)
z = y.cpu() + x.cpu()

net = nn.Linear(3, 1)
list(net.parameters())[0].device

net.cuda()
list(net.parameters())[0].device

x = torch.rand(2, 3).cuda()
print(net(x))

