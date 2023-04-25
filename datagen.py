import pandas as pd
import numpy as np
import argparse
from dataloader import error_item, groundY 

NUMBER_OF_COMMON_THRES = 10

def datagen(args):
    hp_path = args.data_path
    n = args.size_of_train_data
    pnum = args.pnum
    causal_rate = args.causal_rate

    Ydependency = args.Ydependency
    variance_type = args.variance_type
    error_distribution = args.error_distribution
    num_component = args.num_component

    if args.use_covariates:
        c1_coef = args.c1_coef
        c2_coef = args.c2_coef
    else:
        c1_coef, c2_coef = 0, 0
    sep_traintest = args.sep_traintest


    hp = pd.read_csv(hp_path, index_col = 0).to_numpy()


    # generate x set
    train_x, test_x = subset_from_hp(hp, n, pnum, sep_traintest) # from hp, 9/10 as train data， 1/10 as test data. In train data, sample 2*n to form train dataset, test data is test dataset
    num_maf_com = sum(np.mean(train_x, 0) > 0.1)
    while num_maf_com < NUMBER_OF_COMMON_THRES:
        train_x, test_x = subset_from_hp(hp, n, pnum, sep_traintest)
        num_maf_com = sum(np.mean(train_x, 0) > 0.1)

    not_all_0_p_idx = (np.sum(train_x, 0) != 0)
    tidy_train_x = train_x[:,not_all_0_p_idx]
    tidy_test_x = test_x[:, not_all_0_p_idx]
    print(f"x data generated from seed {args.seed}: train dataset {np.shape(tidy_train_x)} matrix; test dataset {np.shape(tidy_test_x)} matrix.")
    print(f"use covariates: {args.use_covariates}")
    


    ## generate corresponding \hat{y} (ground truth) from the x set above
    ntrain, ntest, pdat = np.shape(tidy_train_x)[0], np.shape(tidy_test_x)[0], np.shape(tidy_train_x)[1]
    print(f"n_feature: {pdat}")
    
    c1 = np.random.normal(4,1, n).reshape((n,1))
    c2 = np.random.normal(2,1.5, n).reshape((n,1))
    

    n_causal = round(causal_rate * pdat)
    print(f'n.causal = {n_causal}')
    causal_p_idx = np.random.choice(range(pdat),n_causal, replace = False)

    beta_true = np.random.uniform(0,1,n_causal).reshape((n_causal,1))
    if Ydependency == 'id':
        Ydependency_function = lambda xbeta: xbeta
    elif Ydependency == 'power2':
        Ydependency_function = lambda xbeta: np.square(xbeta)
    elif Ydependency == 'sqrt':
        Ydependency_function = lambda xbeta: np.sqrt(xbeta)

    train_y_true = Ydependency_function(np.matmul(tidy_train_x[:, causal_p_idx], beta_true)+ c1*c1_coef + c2*c2_coef).reshape(-1)
    test_y_true = Ydependency_function(np.matmul(tidy_test_x[:, causal_p_idx], beta_true)).reshape(-1)

    ## adding error term
    ## recommended: num_component <= 3
    ### determine the loc&scale&weight of components

    loc_ls = np.random.uniform(low = -num_component, high = num_component, size = num_component)
    scale_ls = np.random.uniform(low = 0.5, high = num_component, size = num_component)
    weight_ls = np.random.uniform(low = 0, high = 1, size = num_component)
    weight_ls = weight_ls/np.sum(weight_ls)

    train_y = [] # will be ls of groundY obj, while train_y_true is ls of numeric
    for y in train_y_true:
        if variance_type == 'homo':
            error_item_list = [error_item(err_type = error_distribution, loc = obj[0], scale = obj[1], weight = obj[2]) for obj in zip(loc_ls, scale_ls, weight_ls)]
        elif variance_type == 'hete':
            error_item_list = [error_item(err_type = error_distribution, loc = obj[0]*(1+0.5*y), scale = obj[1]*(1+0.5*y), weight = obj[2]) for obj in zip(loc_ls, scale_ls, weight_ls)]
        gy = groundY(y, error_item_list)
        train_y.append(gy)

    test_y = [] # will be ls of groundY obj
    for y in test_y_true:
        if variance_type == 'homo':
            error_item_list = [error_item(err_type = error_distribution, loc = obj[0], scale = obj[1], weight = obj[2]) for obj in zip(loc_ls, scale_ls, weight_ls)]
        elif variance_type == 'hete':
            error_item_list = [error_item(err_type = error_distribution, loc = obj[0]*(1+0.5*y), scale = obj[1]*(1+0.5*y), weight = obj[2]) for obj in zip(loc_ls, scale_ls, weight_ls)]
        gy = groundY(y, error_item_list)
        test_y.append(gy)
    


    ## split train into train + dev
    size_of_dev = round(n*0.1)

    out_dev_x = tidy_train_x[range(size_of_dev), :]
    out_train_x = tidy_train_x[range(size_of_dev,n), :]
    out_test_x = tidy_test_x

    out_dev_y = train_y[0:size_of_dev]
    out_train_y = train_y[size_of_dev:n]
    out_test_y = test_y
    
    return out_train_x, out_train_y, out_dev_x, out_dev_y, out_test_x, out_test_y, pdat



def subset_from_hp(hp, n, p, sep_traintest = False):
    nhp, mhp = np.shape(hp)
    ntest = round(nhp / 20)*2
    if sep_traintest:
        ntrain = nhp - ntest
    else:
        ntrain = nhp
    # maf = np.sum(hp, 0) / nhp
    idx_shuffled = np.random.choice(range(nhp), nhp, replace = False)
    idx_train = idx_shuffled[:ntrain]
    idx_test = idx_shuffled[-ntest:]
    index_start = np.random.choice(range(mhp - p), 1)[0]
    if n > nhp / 2:
        n_index = np.random.choice(idx_train, 2*n, replace = True)
    else:
        n_index = np.random.choice(idx_train, 2*n, replace = False)
    train_x = hp[n_index[range(n)], index_start:(index_start + p)] + hp[n_index[range(n,2*n)], index_start:(index_start + p)]
    test_x = hp[idx_test[range(int(ntest/2))], index_start:(index_start + p)] + hp[idx_test[range(int(ntest/2), ntest)], index_start:(index_start + p)] 

    return train_x, test_x

if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_seed = 42
    args.data_path = 'C:/Users/V1nce/Desktop/thesis/code/hp.csv'
    args.size_of_train_data = 150
    args.pnum = 50
    args.causal_rate = 0.3

    args.Ydependency = 'sqrt'
    args.num_component = 2
    args.variance_type = 'homo'
    args.error_distribution = 'normal'

    args.c1_coef = 1.2
    args.c2_coef = 0.4
    args.use_covariates = False
    args.sep_traintest = False
    
    test_set = datagen(args)
    print(f"datagen() function run with test args. Output data size as follows:")
    print(f"size of x: {np.shape(test_set[0])}, {np.shape(test_set[2])}, {np.shape(test_set[4])}")
    print(f"size of y: {len(test_set[1])}, {len(test_set[3])}, {len(test_set[5])}")
    print(f'dev_y_observed: {[obj.observed() for obj in test_set[3]]}')
    print('-----------------------------')
    print(f"Example of y_loss: y_exp = test_set[3][1] # an groundY obj")
    y_exp = test_set[3][1]
    print(f"y_exp.true_y: {y_exp.true_y}")
    print(f"y_exp.observed(): {y_exp.observed()}")
    print(f"y_exp.num_of_component: {y_exp.num_of_component}")
    print(f"y_exp.weights: {y_exp.weights}")

    ## example of loss
    GMMparam = (torch.tensor([[0.1987, 0.1920, 0.1730, 0.2136, 0.2228]]), torch.tensor([[1.3289, 1.1590, 1.1179, 1.1215, 1.3782]]), torch.tensor([[ 0.1752, -0.3880, -0.1317, -0.2658, -0.1357]])) # pi, sigma, mu
    print(f"y_exp.MLEloss(GMMparam): {y_exp.MLEloss(GMMparam)}")
    print(f"y_exp.KLloss(GMMparam): {y_exp.KLloss(GMMparam)}")

    


