import argparse 
import torch
from torch import nn
from tqdm import tqdm

from datagen import datagen
from dataloader import GetSKATDataset, SKATCollateFn, EarlyStop # ./dataloader.py
from model import GMM
from train import train, test

import math
import numpy as np
import random
from scipy import integrate
import scipy.stats as st
from scipy.optimize import fsolve
from torch.utils.data import DataLoader

from sklearn.model_selection import ParameterGrid
from load_default_param import load_default_param

args = load_default_param()

NUM_OF_TEST_PER_SETTING = 75
BASE_SEED = 5

torch.backends.cudnn.deterministic = True  # cudnn


args.sep_traintest = True
args.size_of_train_data = 1000
args.pnum = 500
args.causal_rate = 0.2
args.seed = BASE_SEED

# param_grid = {'Ydependency': ['id'], 
#               'variance_type': ['hete'],
#               'error_distribution': ['normal'],
#               'num_component': [1]}
# param_dict = ParameterGrid(param_grid)

args.total_steps = 15000
args.gradient_accumulate_steps = 10
args.early_stop_test_steps = 500
args.early_stop_step_tol = 2
args.learning_rate = 2e-3

setting = {'Ydependency': 'id', 'error_distribution': 'normal', 'num_component': 1, 'variance_type': 'hete'}
args.Ydependency = setting['Ydependency']
args.variance_type = setting['variance_type']
args.error_distribution = setting['error_distribution']
args.num_component = setting['num_component']


torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
np.random.seed(args.seed)  # numpy
torch.backends.cudnn.deterministic = True  # cudnn
train_x, train_y, dev_x, dev_y, test_x, test_y, pdat = datagen(args)
train_dataset = GetSKATDataset(train_x, train_y)
# model = GMM()
dev_dataset =  GetSKATDataset(dev_x, dev_y)
test_dataset = GetSKATDataset(test_x, test_y)
train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, collate_fn= SKATCollateFn)
dev_dataloader = DataLoader(dev_dataset, args.batch_size, shuffle = True, collate_fn= SKATCollateFn)
test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle = True, collate_fn= SKATCollateFn)

model = GMM(n_feature = pdat, KMIX = args.kmix)
model.to(args.device)

test_KLloss = train(args, model, train_dataloader, dev_dataloader, None)


def get_q(model, dataloader):
    print(f'-------------------- q start --------------------')
    model.eval()
    loss = []
    with torch.no_grad():
        steps = 0
        bar = tqdm(dataloader)
        for _, batch in enumerate(bar):
            steps += 1
            # note: batch_size is fixed 1
            x = batch[0].to(args.device)
            y = batch[1][0]
            out_param = model(x)
            steploss = y.Wloss(out_param)
            if steploss != None:
                loss.append(steploss)
            bar.set_description("Step: {}".format(steps))
    
    model.train()
    print(f'length of q: {len(loss)}/{steps}')
    print(f'Wloss mean(std): {np.mean(loss):8.5f}({np.std(loss):8.5f})')
    print(f'--------------------- q end ---------------------')
    return loss
 
def set_message(message):
    with open('temp_result.txt', 'a', encoding = 'utf-8') as f:
        f.write(message)
        
qhat = get_q(model, train_dataloader)


import warnings
def get_q_quantile(q, upper_alpha):
    if len(q) < 1/upper_alpha:
        warnings.warn('length of q too short, quantile estimation might be unreliable')
    idx = max(1, int(upper_alpha * len(q)))
    return np.sort(q)[-idx]

def Wloss(GMMparam1, GMMparam2):
    p1, s1, m1 = GMMparam1
    p2, s2, m2 = GMMparam2
    if isinstance(p1, np.ndarray):
        p1, s1, m1 = p1.tolist()[0], s1.tolist()[0], m1.tolist()[0]
        p2, s2, m2 = p2.tolist()[0], s2.tolist()[0], m2.tolist()[0]
    elif isinstance(p1, list):
        p1, s1, m1 = p1[0], s1[0], m1[0]
        p2, s2, m2 = p2[0], s2[0], m2[0]

    p1 = [pi/sum(p1) for pi in p1]
    p2 = [pi/sum(p2) for pi in p2]

    # integration range
    a = min(m1+m2) - 50*max(s1+s2)
    b = max(m1+m2) + 50*max(s1+s2)
    
    def absdifffun(y): # = predicted
        f1 = sum([pi*st.norm.cdf(y, loc = mu, scale = sigma) for pi, sigma, mu in zip(p1, s1, m1)])
        f2 = sum([pi*st.norm.cdf(y, loc = mu, scale = sigma) for pi, sigma, mu in zip(p2, s2, m2)])
        return abs(f1-f2)
    
    l = integrate.quad(absdifffun,a,b)[0]
    return l

# p,s,m
# Wloss([[[0.5,0.5]],[[2,1]],[[3,1]]],[[[0.5,0.5]],[[2+2.907,1]],[[3,1]]])

def get_param_range(q_alpha, GMMparam):
    # shape of param, e.g. KMIX = 3: [[[\pi_1, \pi_2, \pi_3]], [[\sigma_1, \sigma_2, \sigma_3]], [[\mu_1, \mu_2, \mu_3]]]
    # output: [[[\pi_1l, \pi_2l, \pi_3l], [\pi_1r, \pi_2r, \pi_3r]], [[etcl],[etcr]], [[etcl], [etcr]]]

    # q_alpha = get_q_quantile(q, alpha)
    DELTA_D_PER_SIGMA = 0.7978845


    # conformal measure: W distance with p = 1

    p, s, m = GMMparam
    if isinstance(p, np.ndarray):
        p, s, m = p.tolist()[0], s.tolist()[0], m.tolist()[0]
    elif isinstance(p, list):
        p, s, m = p[0], s[0], m[0]

    result = []
    # deal with \pi: (not rigorous here)
    def difffun(deltapi, idx):
        param1 = [[[i for i in p]],[[i for i in s]],[[i for i in m]]]
        param2 = [[[i for i in p]],[[i for i in s]],[[i for i in m]]]
        newp = param1[0][0]
        # if newp[idx] + deltapi <= 0:
        #     newp[idx] = 0 # i.e. when reaching the boundary, the function value keeps const and wait until fslove() stop optim.
        # else:
        
        param2[0][0][idx] += deltapi
        param2[0][0] = [pii/(1+deltapi) for pii in param2[0][0]]
        return Wloss(param1,param2) - q_alpha
    
    result_p_l = []
    result_p_r = []
    
    for idx in range(len(p)):
        print(fsolve(lambda x: difffun(x,idx), -1e-3))
        result_p_l.append(max(0, p[idx] + fsolve(lambda x: difffun(x,idx), -1e-3).tolist()[0]))
        result_p_r.append(p[idx] + fsolve(lambda x: difffun(x,idx),  1e-3).tolist()[0])
    result.append([result_p_l, result_p_r])
    # deal with \sigma:
    result.append([[max(0 ,sigmai - q_alpha/(pii*DELTA_D_PER_SIGMA)) for sigmai, pii in zip(s,p)], [sigmai + q_alpha/(pii*DELTA_D_PER_SIGMA) for sigmai, pii in zip(s,p)]])
    # deal with \mu:
    result.append([[mui - q_alpha/pii for mui, pii in zip(m,p)], [mui + q_alpha/pii for mui, pii in zip(m,p)]])
    

    return result


rg = get_param_range(1, [[[0.4,0.6]],[[2,1]],[[3,1]]])


def GMM_cdf_function(y, GMMparam):
    p, s, m = GMMparam
    if isinstance(p, np.ndarray):
        p, s, m = p.tolist()[0], s.tolist()[0], m.tolist()[0]
    elif isinstance(p, list):
        p, s, m = p[0], s[0], m[0]
    p = [pi/sum(p) for pi in p]
    return sum([pi*st.norm.cdf(y, loc = mu, scale = sigma) for pi, sigma, mu in zip(p, s, m)])


import scipy

scipy.optimize.minimize(lambda para, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)


import math
import matplotlib
np.histogram(qhat)

list([[1,2],[2,3]])

GMMparam_tolist = lambda x: [y for l in x for y in GMMparam_tolist(l)] if type(x) is list else [x]
GMMparam_tostack = lambda x: [[x[0:int(len(x)/3)]],[x[int(len(x)/3):int(len(x)*2/3)]],[x[int(len(x)*2/3):len(x)]]]

def get_bound(out):
    return (list(zip(out[0][0], out[0][1])) + list(zip(out[1][0], out[1][1])) + list(zip(out[2][0], out[2][1])))

get_bound(rg)

def fun(y):
    return lambda param: GMM_cdf_function(y, GMMparam_tostack(param))

def con(model_GMM_param, q):
    return lambda param: q - Wloss(model_GMM_param, GMMparam_tostack(param))



# p,s,m
mGMM = [[[0.5,0.5]],[[2,1]],[[3,1]]]
bound = get_bound(rg)
cons = ({'type': 'ineq', 'fun': con(mGMM, 1)})


scipy.optimize.minimize(fun(3), method = 'SLSQP', x0 =  GMMparam_tolist(mGMM), bounds = bound, constraints = cons)
GMM_cdf_function(3, mGMM)

Wloss(mGMM, GMMparam_tostack([1.04275445e+00, 4.12833461e-01, 6.63507113e-15, 6.25536210e-01,3.00003668e+00, 6.07902950e-02]))

