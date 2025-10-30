import torchax
from transformers import BertForMaskedLM
import torch
from torch.utils.data import DataLoader, Dataset


from torchax.interop import JittableModule, jax_value_and_grad

import torchax
torchax.enable_globally()

class XORDataset(Dataset):
    def __init__(self, num: int = 1000):
        self.data = torch.randn(num, 2)
        self.label = self.data.sum(axis= -1, keepdim = True) == 1
        self.num  = num


    def __len__(self, ):
        return self.num

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


dataset = XORDataset()


class SuperLinear(torch.nn.Module):
    def __init__(self, params: list[int] = [2, 64, 1]):
        super().__init__()
        self.linear1 = torch.nn.Linear(params[0], params[1]) 
        self.linear2 = torch.nn.Linear(params[1], params[2]) 
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        o = self.linear1(x)
        o = torch.nn.functional.tanh(o)
        o = self.dropout(o)
        print(f"DEBUGPRINT[51]: interop.py:39: o={o}")
        o = self.linear2(o)
        return o

model = SuperLinear().to("jax")

weights = model.state_dict()

# model = JittableModule(model)

eps = 1e-8

def loss_fn(weight, batch):
    x, y = batch
    out: torch.Tensor =  torch.func.functional_call(model, weights, x) 
    print(f"DEBUGPRINT[50]: interop.py:50: out={out}")
    loss = - (y * torch.log(out+ eps) + (1-y) * torch.log(1 - out + eps)).mean()
    return loss


dataloader=  DataLoader(dataset, batch_size = 10)
data = next(iter(dataloader))
print(f"DEBUGPRINT[48]: interop.py:51: data={data}")
grad_fn = jax_value_and_grad(loss_fn) 

data = (data[0].to("jax"), data[1].to("jax"))
print(data[0])
loss, grad = grad_fn(weights, data)
print(f"DEBUGPRINT[47]: interop.py:52: loss={loss}")
