import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import random
from scipy import integrate
import scipy.stats as st
import math

import json
import os
import sys
import re







class GetSKATDataset(Dataset):
    def __init__(self, dat_x, dat_y):
        super().__init__()
        self.X = dat_x # [numeric vector]
        self.Y = dat_y # [groundY obj]
        self.n_feature = self.X.shape[1]
    
    def __getitem__(self, index):
        assert index < self.__len__()
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


def SKATCollateFn(batch):
    batch = list(zip(*batch))
    x = torch.tensor(batch[0], dtype=torch.float)
    y = batch[1]
    return x,y



class error_item():
    def __init__(self, err_type = 'normal', loc = 0, scale =1, weight = 1):
        assert err_type in ['normal','cauchy','chisq']
        self.err_type = err_type
        self.loc = loc
        self.scale = scale
        self.weight = weight
    def err_rvs(self):
        if self.err_type == 'normal':
            r2 = st.norm.rvs(size = 1, loc = self.loc, scale = self.scale)
        elif self.err_type == 'cauchy':
            r2 = st.cauchy.rvs(size = 1, loc = self.loc, scale = self.scale)
        elif self.err_type == 'chisq':
            r2 = st.chi2.rvs(size = 1, loc = self.loc, scale = self.scale, df = 1) # note: non-zero mean
        return r2[0]

    def pdf(self, y, yt):
        if self.err_type == 'normal':
            p = st.norm.pdf(y, loc = yt + self.loc, scale = self.scale)
        elif self.err_type == 'cauchy':
            p = st.cauchy.pdf(y, loc = yt + self.loc, scale = self.scale)
        elif self.err_type == 'chisq':
            p = st.chi2.pdf(y, loc = yt + self.loc, scale = self.scale, df = 1) # note: non-zero mean
        return p

class groundY():
    def __init__(self, true_y, error_item_list): # obj in error_item_list should all be instance of error_item() class
        super().__init__()
        self.true_y = true_y 
        self.num_of_component = len(error_item_list)
        self.error_item_list = error_item_list
        # check sumation normalization of weights
        weights = [obj.weight for obj in error_item_list]
        self.weights = [weight/sum(weights) for weight in weights]
        for i in range(len(error_item_list)):
            self.error_item_list[i].weight = self.weights[i]

        self.loc = [obj.loc for obj in error_item_list]
        self.scale = [obj.scale for obj in error_item_list]

        # create observed value
        which = np.random.choice(range(self.num_of_component), size = 1, p = self.weights)[0]
        self.rvs = self.true_y + self.error_item_list[which].err_rvs()


    def observed(self):# fixed if .observed() 
        return self.rvs
   

    # MLEloss is used in pytorch network, back prop needed so must use torch function. KLloss is used for val, so just use numpy
    def MLEloss(self, GMMparam, epsilon = 1e-9):
        if self.rvs is not None: 
            y = torch.tensor(self.rvs)
            out_pi, out_sigma, out_mu = GMMparam
            expand_y = y.reshape((-1,1)).tile((1,out_pi.size()[1]))
            result = torch.sum(out_pi/out_sigma*torch.exp(-((expand_y-out_mu)/out_sigma)**2/2), dim = 1, keepdim=True) + epsilon
            return torch.mean(-torch.log(result))

        else:
            print('MLEloss available only after .observed()')
            return 0
    
    def KLloss(self, GMMparam, epsilon = 1e-9):
        # note: input ONE dataitem
        p, s, m = GMMparam
        p, s, m = p.tolist()[0], s.tolist()[0], m.tolist()[0]
        try:
            # integration range
            a = min(self.loc + m)-6*max(self.scale + s)
            b = max(self.loc + m)+6*max(self.scale + s)
            
            def qfun(y): # = predicted
                return sum([pi*st.norm.pdf(y, loc = mu, scale = sigma) for pi, sigma, mu in zip(p, s, m)])
            def pfun(y): # = ground truth
                return sum([err.weight*err.pdf(y,self.true_y) for err in self.error_item_list])
            def plogpq(y):
                return pfun(y)*(math.log(pfun(y)+epsilon)-math.log(qfun(y)+epsilon))
            
            l = integrate.quad(plogpq,a,b) # l[0] = integration, l[1] = calculation error
            return l[0]
        except:
            print('Integration raised math error; skipped.')
            with open('wrong_integ.txt', 'w', encoding = 'utf-8') as f:
                f.write(f"y_true: {self.true_y}; error: {[(err.loc, err.scale, err.weight) for err in self.error_item_list]}\n\t GMMparam: {GMMparam}")
            return None
        
    def Wloss(self, GMMparam): # Wloss with p = 1
        p, s, m = GMMparam
        p, s, m = p.tolist()[0], s.tolist()[0], m.tolist()[0]
        try:
            # integration range
            a = self.rvs - 50*max(s)
            b = self.rvs + 50*max(s)
            
            def leftfun(y): # = predicted
                return sum([pi*st.norm.cdf(y, loc = mu, scale = sigma) for pi, sigma, mu in zip(p, s, m)])
            def rightfun(y): # = predicted
                return 1 - sum([pi*st.norm.cdf(y, loc = mu, scale = sigma) for pi, sigma, mu in zip(p, s, m)])
            
            l = integrate.quad(leftfun,a,self.rvs)[0] + integrate.quad(rightfun,self.rvs,b)[0]
            return l

        except:
            print('Integration raised math error; skipped.')
            with open('wrong_integ.txt', 'w', encoding = 'utf-8') as f:
                f.write(f"y_true: {self.true_y}; error: {[(err.loc, err.scale, err.weight) for err in self.error_item_list]}\n\t GMMparam: {GMMparam}")
            return None




class EarlyStop():
    def __init__(self, step_tol = 3, delta = 0, num_test_after_stop = 10):
        self.step_tol = step_tol
        self.delta = delta
        self.num_test_after_stop = num_test_after_stop

        self.best_MLEloss = None
        self.KL_at_best_MLEloss = None
        self.do_stop = False
        self.metric_increase_counter = 0
        self.after_stop_counter = 0
        
    def __call__(self, MLEloss, KLloss, args, model):
        if self.after_stop_counter == 0: # have not reached early stop: test on dev for early stopping moment
            if self.best_MLEloss is None:
                self.best_MLEloss = MLEloss
                self.KL_at_best_MLEloss = KLloss
                # torch.save(model, os.path.join(args.output_dir, 'best_model.pth'))
            elif (MLEloss > self.best_MLEloss - self.delta):
                self.metric_increase_counter += 1
                if self.metric_increase_counter >= self.step_tol:
                    self.after_stop_counter = 1 # count self.num_test_after_stop after early stop to ensure behaviour of model
            else:
                self.best_MLEloss = MLEloss
                self.KL_at_best_MLEloss = KLloss
                # torch.save(model, os.path.join(args.output_dir, 'best_model.pth'))
                self.metric_increase_counter = 0
        else: # i.e. if self.after_stop_counter != 0
            if self.after_stop_counter >= self.num_test_after_stop:
                self.do_stop = True
            else:
                self.after_stop_counter += 1

        return self.do_stop


        

