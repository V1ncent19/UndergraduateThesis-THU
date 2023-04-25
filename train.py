import torch
from torch import nn
from dataloader import GetSKATDataset, SKATCollateFn, EarlyStop, error_item, groundY # ./dataloader.py
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from model import GMM
import numpy as np
import scipy.stats as st
from tqdm import tqdm
import wandb


def train(args, model, dataloader, dev_dataloader, test_dataloader):
    if args.use_wandb:
        wandb.init(name = args.wandb_name, project = 'GMM_Sweep2', entity="v1ncent19")
        wandb.config.update(args)
        wandb.watch(model)

    early_stop_manager = EarlyStop(step_tol = args.early_stop_step_tol)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)


    # cosine decay
    if args.warmup_steps == -1:
        args.warmup_steps = int(args.total_steps/10)
    if args.use_cos_schedule == True:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = args.total_steps
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps = args.warmup_steps
        )


    print(f'======================== Starts training, total steps {args.total_steps} ========================')

    global_step = 0
    model.train()
    while global_step <= args.total_steps:
        bar = tqdm(dataloader)
        print(f'Running dataloader of size {len(bar)}:')
        for _, batch in enumerate(bar):
            global_step += 1
            # note: batch_size is fixed 1
            x = batch[0].to(args.device)
            y = batch[1][0]
            out_param = model(x)
            loss = y.MLEloss(out_param)
            loss.backward() # append grad
            scheduler.step()
            
            if (global_step + 1) % args.gradient_accumulate_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()
            
            
            step_log = {'MLEloss':loss, 'lr':optimizer.state_dict()['param_groups'][0]['lr']}
            bar.set_description("Step: {}, MLEloss: {:.4f}".format(global_step, loss.item()))
            

            if global_step % args.early_stop_test_steps == 0:
                dev_MLEloss = dev(args, model, dev_dataloader)
                # test_KLloss = test(args, model, test_dataloader)

                step_log['dev_MLEloss'] = dev_MLEloss
                # step_log['test_KLloss'] = test_KLloss

                if early_stop_manager(dev_MLEloss, -1, args, model):
                    print(f'early stop at step {global_step}.')
                    global_step = args.total_steps + 1

            if args.use_wandb:
                wandb.log(step_log)
            
            if global_step > args.total_steps:
                break
                
            
    if args.use_wandb:
        wandb.run.summary['best_dev_MLEloss'] = early_stop_manager.best_MLEloss
        wandb.run.summary['final_test_KLloss'] = early_stop_manager.KL_at_best_MLEloss

    if test_dataloader is not None:
        test_KLloss = test(args, model, test_dataloader)
        return test_KLloss

    # print('=========== Calculating KLloss on test set ============')
    # test_loss = test(args, model, test_dataloader)
    # print("KLloss on test dataset: {:.3f}".format(test_loss))
    # print('=======================================================')


def dev(args, model, dataloader):
    print(f'-------------------- dev start --------------------')
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
            loss.append(y.MLEloss(out_param))
            bar.set_description("Step: {}".format(steps))
    model.train()
    print(f'dev MLEloss: {sum(loss)/len(loss)}')
    print(f'--------------------- dev end ---------------------')
    return sum(loss)/len(loss)

def test(args, model, dataloader):
    print(f'-------------------- test start --------------------')
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
            steploss = y.KLloss(out_param)
            if steploss != None:
                loss.append(steploss)
            bar.set_description("Step: {}".format(steps))
    
    model.train()
    print(f'test KLloss: {sum(loss)/len(loss)}')
    print(f'--------------------- test end ---------------------')
    return sum(loss)/len(loss)
 