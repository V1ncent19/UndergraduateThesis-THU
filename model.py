import torch
from torch import nn
from dataloader import GetSKATDataset, SKATCollateFn
from torch.utils.data import DataLoader

class GMM(nn.Module):
    def __init__(self, n_feature, KMIX, use_dropout = False, dropout_rate = 0.5) -> None:
        super().__init__()
        self.KMIX = KMIX
        self.linear1 = nn.Sequential(nn.Linear(n_feature, 8), nn.ReLU())        
        self.linear2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        self.linear4 = nn.Linear(8, 3*KMIX)
        self.use_dropout = use_dropout
        if self.use_dropout == True:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x):
        out1 = self.linear1(x)
        if self.use_dropout == True:
            out1 = self.dropout1(out1)
        out2 = self.linear2(out1)
        if self.use_dropout == True:
            out2 = self.dropout2(out2)
        out3 = self.linear3(out2)
        if self.use_dropout == True:
            out3 = self.dropout3(out3)
        out4 = self.linear4(out3)       
        GMMparam = get_mixture_coeff(out4, self.KMIX)
        return GMMparam

def get_mixture_coeff(linear_out, KMIX):
    out_pi, out_sigma, out_mu = torch.split(linear_out, KMIX, dim = 1)
    max_pi =torch.max(linear_out, 1, keepdim = True).values
    exp_pi = torch.exp(torch.subtract(out_pi, max_pi))
    nor_pi = torch.nn.functional.normalize(exp_pi.float(), p=1, dim= 1)
    nor_sigma = torch.exp(out_sigma)
    return nor_pi, nor_sigma, out_mu

if __name__ == '__main__':
    print('testing get_mixture_coeff()')
    lr_out = torch.tensor([[3,5,8,3,5,2],[3,4,7,2,6,4],[3,6,2,7,5,3]], dtype=torch.float32)
    print(f'linear out = {lr_out}, kmix = 2')
    print(f'result: {get_mixture_coeff(lr_out, 2)}')
