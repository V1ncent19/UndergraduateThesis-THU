import argparse 
import torch
from torch import nn


from datagen import datagen
from dataloader import GetSKATDataset, SKATCollateFn, EarlyStop # ./dataloader.py
from model import GMM
from train import train, test

import numpy as np
import scipy.stats as st
from tqdm import tqdm #
from torch.utils.data import DataLoader
import os
import wandb




def load_default_param():
    parser = argparse.ArgumentParser()
    ## result recording
    parser.add_argument(
        "--output_dir",
        default = "C:/Users/V1nce/Desktop/thesis/code/results",
        type = str,
        help = "result statistics record dir"
    )
    # parser.add_argument(
    #     "--use_wandb",
    #     action = "store_true",
    #     help = "use wandb to record results"
    # )
    # Defualt set as True
    parser.add_argument(
        "--wandb_name",
        default = "GMMtest",
        type = str,
        help = "name of the run"
    )



    ## data generation param
    parser.add_argument(
        "--seed",
        default = 42,
        type = int,
        help = "random gen seed"
    )

    parser.add_argument(
        "--data_path",
        default = 'C:/Users/V1nce/Desktop/thesis/code/hp.csv',
        type = str,
        help = "hp.csv path"
    )
    parser.add_argument(
        "--sep_traintest",
        action = "store_true",
        help = "If separate train set and test set"
    )
    parser.add_argument(
        "--size_of_train_data",
        default = 2000,
        type = int,
        help = "n of train_x"
    )
    # parser.add_argument(
    #     "--size_of_test_data",
    #     default = None,
    #     help = "n of test_x"
    # )
    parser.add_argument(
        "--pnum",
        default = 100,
        type = int,
        help = "Number of sites to sample from hp"
    )
    parser.add_argument(
        "--causal_rate",
        default = 0.1,
        type = float,
        help = "ratio of sites with non-zero beta"
    )

    ## exp
    parser.add_argument(
        "--Ydependency",
        default = 'id',
        choices=["id", "power2", "sqrt"], 
        help = "Y dependency mode on X\\beta"
    )
    parser.add_argument(
        "--variance_type",
        default = 'homo',
        choices=["homo", "hete"], 
        help = "type of variance"
    )
    parser.add_argument(
        "--error_distribution",
        default = 'normal',
        choices=["normal", "chisq", "cauchy"], 
        help = "distribution of error",
    )
    parser.add_argument(
        "--num_component",
        default = 1,
        type = int,
        help = "number of component in error",
    )
    ## exp end


    parser.add_argument(
        "--use_covariates",
        action = "store_true",
        help = "If use covariates in datagen"
    )
    parser.add_argument(
        "--c1_coef",
        default = 1.2,
        type = float,
        help = "covariates influence coef"
    )
    parser.add_argument(
        "--c2_coef",
        default = 0.4,
        type = float,
        help = "covariates influence coef"
    )

    ## model param
    parser.add_argument(
        "--kmix",
        default = 5,
        type = int,
        help = "number of mixture components in GMM"
    )
    ## training param
    parser.add_argument(
        "--total_steps",
        default = 1000,
        type = int,
        help = "total training steps."
    )
    parser.add_argument(
        "--device",
        default = 'cpu',
        choices=["cpu", "cuda:0"], 
        help = 'device for training'
    )
    parser.add_argument(
        "--gradient_accumulate_steps",
        default = 5,
        type = int,
        help = "steps before gradient accumulation, or equivalently batch size"
    )
    # args.batch_size is set 1
    parser.add_argument(
        "--early_stop_step_tol",
        default = 2,
        type = int,
        help = "Stop when step_tol times of f1 drop on dev"
    )
    parser.add_argument(
        "--early_stop_test_steps",
        default = 6000,
        type = int,
        help = "steps to test early stop"
    )
    parser.add_argument(
        "--learning_rate",
        default = 1e-3,
        type = float,
        help = "learning rate"
    )
    # parser.add_argument(
    #     "--use_cos_schedule",
    #     action = "store_true",
    #     help = "use cosine schedule decay or not"
    # )
    # now default set as True
    parser.add_argument(
        "--warmup_steps",
        default = -1,
        type = int,
        help = "steps for warmup before cosine decay"
    )
    # parser.add_argument(
    #     "--schedule", 
    #     type=str, 
    #     default="constant",
    #     choices=["linear", "cosine", "constant"], 
    #     help="Schedule."
    # )
    parser.add_argument(
        "--max_grad_norm", 
        type=float, 
        default= 1.0,
        help="avoid gradient explosion"
    )
    args = parser.parse_args()

    args.use_cos_schedule = True
    args.use_wandb = False
    args.batch_size = 1
    args.device = 'cpu'
    base_data_dir = 'C:/Users/V1nce/Desktop/thesis/code'

    return args