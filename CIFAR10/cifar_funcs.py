import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
import ipdb
import random
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

import time

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]    

'''
#DEFAULTS
pgd_linf: epsilon=0.03, alpha=0.003, num_iter = 40
pgd_l0  : epsilon = 12, alpha = 1
pgd_l1_topk  : epsilon = 12, alpha = 0.05, num_iter = 40, k = rand(5,20) --> (alpha = alpha/k *20)
pgd_l2  : epsilon =0.5, alpha=0.05, num_iter = 50

'''

def pgd_l2(model, X, y, epsilon=0.5, alpha=0.05, num_iter = 50, device = "cuda:0", restarts = 0, version = 0):
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1 (Test time)
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        if torch.sum(norms(delta.grad.detach()) == 0) != 0:
            print(delta.grad.detach())
            print(norms(delta.grad.detach()) == 0)
            print('t', t)
            print(output.max(1)[1])
            print(y)
            print(correct)
            print(torch.finfo(torch.float16))
            raise('Norm is 0 again')
        delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
        if torch.sum(norms(delta.grad.detach()) == 0) != 0:
            # print(delta.data)
            raise('Norm is 0 again')
        delta.data *=  epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.data =   torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]     
        delta.grad.zero_()  

    max_delta = delta.detach()

    for _ in range (restarts):
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)*epsilon 
        delta.data /= norms(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data +=  correct*alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.grad.zero_()  

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    return max_delta    


def pgd_l1_topk(model, X,y, epsilon = 12, alpha = 1, num_iter = 20, k = 20, device = "cuda:1", restarts = 1, version = 0, l1_topk_mod = 'normal'):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    # l1_topk_mod: normal (the same as baseline), random_k, adaptive_step, no
    gap = 0
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)

    for t in range (num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        if l1_topk_mod in ['normal', 'random_k']:
            k = random.randint(80,99)
        # alpha = 0.05/k*20
        # print(delta.grad.detach().device)
        # print(delta.grad.detach())
        if torch.sum(norms(delta.grad.detach()) == 0) != 0:
            print(norms(delta.grad.detach()) == 0)
            print('t', t)
            print(output)
            print(y)
            raise('Norm is 0 again')
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, k, l1_topk_mod=l1_topk_mod)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
            # print(norms_l1(delta.data))
            # print(norms_l1(delta.data) > epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_()

    max_delta = delta.detach()

    #Restarts    
    for _ in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            if l1_topk_mod in ['normal', 'random_k']:
                k = random.randint(80,99)
            # alpha = 0.05/k*20
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X,k, l1_topk_mod=l1_topk_mod)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    # print("norms_l0: ", norms_l0(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # print("norms_l1: ", norms_l1(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # input()

    return max_delta

def pgd_l1_top1(model, X,y, epsilon = 12, alpha = 1, num_iter = 20, device = "cuda:1", restarts = 1, version = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    gap = 0
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)

    for t in range (num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        k = 100
        # alpha = 0.05/k*20
        # print(delta.grad.detach().device)
        # print(delta.grad.detach())
        if torch.sum(norms(delta.grad.detach()) == 0) != 0:
            print(norms(delta.grad.detach()) == 0)
            print('t', t)
            print(output)
            print(y)
            raise('Norm is 0 again')
        # top_dir = l1_dir_top1(delta.grad.detach(), delta.data, X)
        delta.data += alpha*correct*l1_dir_top1(delta.grad.detach(), delta.data, X)
        if (norms_l1(delta) > epsilon).any():
            delta.data = proj_l1ball(delta.data, epsilon, device)
            # print(norms_l1(delta.data))
            # print(norms_l1(delta.data) > epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_()

    max_delta = delta.detach()

    #Restarts    
    for _ in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            k = 100
            # alpha = 0.05/k*20
            delta.data += alpha*correct*l1_dir_top1(delta.grad.detach(), delta.data, X)
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_() 
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    # print("norms_l0: ", norms_l0(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # print("norms_l1: ", norms_l1(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # input()

    return max_delta

def pgd_l1_sign_free(model, X, y, epsilon = 12, alpha = 0.05, num_iter = 20, k = 20, device = "cuda:0", restarts = 1, version = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    gap = alpha
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)

    for t in range(num_iter):
        # print(t)
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        step_alpha = alpha
        if torch.sum(norms(delta.grad.detach()) == 0) != 0:
            print(delta.grad.detach())
            print(norms(delta.grad.detach()).squeeze(1).squeeze(1).squeeze(1))
            print((norms(delta.grad.detach()) == 0).squeeze(1).squeeze(1).squeeze(1))
            raise('Norm is 0 again')
        delta.data += step_alpha * correct * delta.grad.detach() / norms(delta.grad.detach())
        if (norms_l1(delta) > epsilon).any():
            # input('check proj_l1ball in pgd_l1_topk')
            delta.data = proj_l1ball(delta.data, epsilon, device)
            # print(norms_l1(delta) > epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_()

    max_delta = delta.detach()

    #Restarts    
    for _ in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            step_alpha = alpha
            delta.data += step_alpha * correct * delta.grad.detach() / norms(delta.grad.detach())
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_()
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    # print('done')
    # input()
    # print("norms_l0: ", norms_l0(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # print("norms_l1: ", norms_l1(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # input()

    return max_delta

def pgd_l1_sign_free_momentum(model, X, y, epsilon = 12, alpha = 0.05, num_iter = 20, k = 20, device = "cuda:0", restarts = 1, version = 0):
    #Gap : Dont attack pixels closer than the gap value to 0 or 1
    gap = alpha
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad = True)
    last_delta = delta.detach()
    momentum = delta - last_delta
    beta = 0.1

    for t in range(num_iter):
        # print(t)
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X+delta), y)
        loss.backward()
        step_alpha = alpha
        if torch.sum(norms(delta.grad.detach()) == 0) != 0:
            print(delta.grad.detach())
            print(norms(delta.grad.detach()).squeeze(1).squeeze(1).squeeze(1))
            print((norms(delta.grad.detach()) == 0).squeeze(1).squeeze(1).squeeze(1))
            raise('Norm is 0 again')
        # delta.data += step_alpha * correct * delta.grad.detach() / norms(delta.grad.detach())
        temp1 = step_alpha * correct * delta.grad.detach() / norms(delta.grad.detach())
        if t == 0:
            momentum = temp1
        else:
            momentum = beta * momentum + temp1 * (1 - beta)
        delta.data += momentum
        if (norms_l1(delta) > epsilon).any():
            proj_delta1 = proj_l1ball(delta.data, epsilon, device)
            mask = (norms_l1(delta) > epsilon).int()
            # print(torch.sum(mask))
        else:
            proj_delta1 = delta
            mask = 0
        proj_delta1 = proj_delta1.detach()
        # delta.data += momentum * (1 - t/(num_iter-1))
        delta.data += momentum
        if (norms_l1(delta) > epsilon).any():
            # input('check proj_l1ball in pgd_l1_topk')
            delta.data = proj_l1ball(delta.data, epsilon, device)
            # print(norms_l1(delta) > epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
        delta.grad.zero_()
        proj_delta2 = delta.detach()
        # momentum = proj_delta2 - last_delta
        # momentum = temp1
        last_delta = delta.detach()

    max_delta = delta.detach()

    #Restarts    
    for _ in range(restarts):
        delta = torch.rand_like(X,requires_grad = True)
        delta.data = (2*delta.data - 1.0)*epsilon 
        delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
        last_delta = delta.detach()
        momentum = delta - last_delta
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only them only for version 1
            loss = nn.CrossEntropyLoss()(model(X+delta), y)
            loss.backward()
            step_alpha = alpha
            # delta.data += step_alpha * correct * delta.grad.detach() / norms(delta.grad.detach())
            temp1 = step_alpha * correct * delta.grad.detach() / norms(delta.grad.detach())
            # if t == 0:
            #     momentum = temp1
            # else:
            #     momentum = beta * momentum + temp1 * (1 - beta)
            delta.data += momentum
            if (norms_l1(delta) > epsilon).any():
                proj_delta1 = proj_l1ball(delta.data, epsilon, device)
                mask = (norms_l1(delta) > epsilon).int()
            else:
                proj_delta1 = delta
                mask = 0
            proj_delta1 = proj_delta1.detach()
            # delta.data += momentum * (1 - t/(num_iter-1))
            # delta.data += momentum
            if (norms_l1(delta) > epsilon).any():
                delta.data = proj_l1ball(delta.data, epsilon, device)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1] 
            delta.grad.zero_()
            proj_delta2 = delta.detach()
            # momentum = proj_delta2 - last_delta
            # momentum = temp1
            last_delta = delta.detach()
        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect] 

    # print('done')
    # input()
    # print("norms_l0: ", norms_l0(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # print("norms_l1: ", norms_l1(max_delta).squeeze(1).squeeze(1).squeeze(1))
    # input()

    return max_delta

def pgd_linf(model, X, y, epsilon=0.03, epsilon_255=8, alpha_sign_free=0.5, num_iter = 10, alpha_255 = 0.8, sign_free = False, device = "cuda:0", restarts = 0, version = 0):
    epsilon = float(epsilon_255) / 255.
    alpha = float(alpha_255) / 255.
    """ Construct FGSM adversarial examples on the examples X"""
    max_delta = torch.zeros_like(X)
    delta = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(X+delta)
        incorrect = output.max(1)[1] != y 
        correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        correct = 1.0 if version == 0 else correct
        #Finding the correct examples so as to attack only them only for version 1
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        if not sign_free:
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        else:
            delta.data = (delta.data + alpha_sign_free*correct*delta.grad.detach()).clamp(-epsilon,epsilon)
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.grad.zero_()
    max_delta = delta.detach()
    
    for _ in range (restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = (delta.data*2.0  - 1.0)*epsilon
        for t in range(num_iter):
            output = model(X+delta)
            incorrect = output.max(1)[1] != y 
            correct = incorrect.logical_not().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            correct = 1.0 if version == 0 else correct
            #Finding the correct examples so as to attack only themonly for version 1
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            if not sign_free:
                delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            else:
                delta.data = (delta.data + alpha_sign_free*correct*delta.grad.detach()).clamp(-epsilon,epsilon)
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,255]
            delta.grad.zero_()

        output = model(X+delta)
        incorrect = output.max(1)[1] != y
        #Edit Max Delta only for successful attacks
        max_delta[incorrect] = delta.detach()[incorrect]   

    return max_delta    


def pgd_l0(model, X,y, epsilon = 12, alpha = 1, num_iter = 0, device = "cuda:1"):
    delta = torch.zeros_like(X, requires_grad = True)
    batch_size = X.shape[0]
    # print("Updated")
    for t in range (epsilon):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        temp = delta.grad.view(batch_size, 1, -1)
        neg = (delta.data != 0)
        X_curr = X + delta
        neg1 = (delta.grad < 0)*(X_curr < 0.1)
        neg2 = (delta.grad > 0)*(X_curr > 0.9)
        neg += neg1 + neg2
        u = neg.view(batch_size,1,-1)
        temp[u] = 0
        my_delta = torch.zeros_like(X).view(batch_size, 1, -1)
        
        maxv =  temp.max(dim = 2)
        minv =  temp.min(dim = 2)
        val_max = maxv[0].view(batch_size)
        val_min = minv[0].view(batch_size)
        pos_max = maxv[1].view(batch_size)
        pos_min = minv[1].view(batch_size)
        select_max = (val_max.abs()>=val_min.abs())
        select_min = (val_max.abs()<val_min.abs())
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = (1-X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max])*select_max
        my_delta[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min]*select_min
        delta.data += my_delta.view(batch_size, 3, 32, 32)
        delta.grad.zero_()
    delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
    
    return delta.detach()

def msd_v0(model, X,y, epsilon_l_1 = 12, alpha_l1 = 1, epsilon_l_2 = 0.5, alpha_l2 = 0.166666, epsilon_l_inf_255 = 8, alpha_linf_255 = 2, num_iter = 10, device = "cuda:0"):
    delta = torch.zeros_like(X,requires_grad = True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_max_loss = torch.zeros(y.shape[0]).to(y.device)
    # alpha_l_1_default = alpha_l_1

    epsilon_linf = float(epsilon_l_inf_255) / 255.
    alpha_linf = float(alpha_linf_255) / 255.
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():          

            # For L1
            delta_l_1 = delta.data + alpha_l1 * delta.grad.detach() / norms(delta.grad.detach())
            if (norms_l1(delta_l_1) > epsilon_l_1).any():
                delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
            delta_l_1 = torch.min(torch.max(delta_l_1.detach(), -X), 1-X) # clip X+delta to [0,1] 

            #For L_2
            delta_l_2  = delta.data + alpha_l2*delta.grad / norms(delta.grad)      
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
            delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]

            #For L_inf
            delta_l_inf=  (delta.data + alpha_linf*delta.grad.sign()).clamp(-epsilon_linf,epsilon_linf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]
            
            #Compare
            delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device)
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data # choose max from three perturbations
            max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss] # choose max in the interation steps
            max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
        delta.grad.zero_()

    return max_max_delta


def pgd_worst_dir(model, X,y, epsilon_l_1 = 12, alpha_l1 = 1, epsilon_l_2 = 0.5, alpha_l2 = 0.166666, epsilon_l_inf_255 = 8, alpha_linf_255 = 2, num_iter = 10, restarts = 0, device = "cuda:0"):
    #Always call version = 0
    # delta_1 = pgd_l1_topk(model, X, y, epsilon = epsilon_l_1, alpha = alpha_l_1,  device = device)
    # delta_2 = pgd_l2(model, X, y, epsilon = epsilon_l_2, alpha = alpha_l_2,  device = device)
    # delta_inf = pgd_linf(model, X, y, epsilon = epsilon_l_inf, alpha = alpha_l_inf, device = device)

    delta_1 = pgd_l1_sign_free(model, X, y, device = device, epsilon = epsilon_l_1, alpha = alpha_l1, restarts=restarts, num_iter=num_iter)
    delta_2 = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2, alpha = alpha_l2, restarts=restarts, num_iter=num_iter)
    delta_inf = pgd_linf(model, X, y, device = device, epsilon_255 = epsilon_l_inf_255, alpha_255 = alpha_linf_255, restarts=restarts, num_iter=num_iter)
    
    batch_size = X.shape[0]

    loss_1 = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_1), y)
    loss_2 = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_2), y)
    loss_inf = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_inf), y)

    delta_1 = delta_1.view(batch_size,1,-1)
    delta_2 = delta_2.view(batch_size,1,-1)
    delta_inf = delta_inf.view(batch_size,1,-1)

    tensor_list = [loss_1, loss_2, loss_inf]
    delta_list = [delta_1, delta_2, delta_inf]
    loss_arr = torch.stack(tuple(tensor_list))
    delta_arr = torch.stack(tuple(delta_list))
    max_loss = loss_arr.max(dim = 0)
    

    delta = delta_arr[max_loss[1], torch.arange(batch_size), 0]
    delta = delta.view(batch_size,3, X.shape[2], X.shape[3])
    return delta


def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad, delta, X, k=20, l1_topk_mod='normal'):
    # l1_topk_mod: normal (the same as baseline), random_k, adaptive_step, no
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]

    grad = grad.detach().cpu().numpy()
    abs_grad = np.abs(grad)
    sign = np.sign(grad)

    max_abs_grad = np.percentile(abs_grad, k, axis=(1, 2, 3), keepdims=True)
    # print(np.percentile(abs_grad, k, axis=(1, 2, 3), keepdims=True)[0][0][0])
    # print(np.percentile(abs_grad, 100 - k, axis=(1, 2, 3), keepdims=True)[0][0][0])
    # input('check')
    tied_for_max = (abs_grad >= max_abs_grad).astype(np.float32)
    if l1_topk_mod in ['normal', 'adaptive_step']:
        num_ties = np.sum(tied_for_max, (1, 2, 3), keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif l1_topk_mod in ['random_k', 'no']:
        l2norm = np.linalg.norm(tied_for_max.reshape(batch_size, -1), axis=1)
        l2norm = np.expand_dims(l2norm, axis=(1,2,3))
        optimal_perturbation = sign * tied_for_max / l2norm
        # l2norm = np.linalg.norm(optimal_perturbation.reshape(batch_size, -1), axis=1)
        # print(l2norm)
        # input('check')


    optimal_perturbation = torch.from_numpy(optimal_perturbation).to(delta.device)
    return optimal_perturbation.view(batch_size, channels, pix, pix)

def l1_dir_top1(grad, delta, X):
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]

    grad = grad.detach().cpu().numpy()
    abs_grad = np.abs(grad).reshape((batch_size, -1))
    sign = np.sign(grad).reshape((batch_size, -1))

    max_abs_grad = np.argmax(abs_grad, axis=1)
    # max_abs_grad = np.stack([np.arange(50), max_abs_grad], axis=1)

    l1_max_dir = np.zeros_like(abs_grad)
    l1_max_dir[np.arange(50), max_abs_grad] = 1
    
    # print(l1_max_dir[:, :15])
    # print(max_abs_grad.shape)
    # input('check l1_dir_top1')

    l1_max_dir *= sign

    optimal_perturbation = torch.from_numpy(l1_max_dir).to(delta.device)
    return optimal_perturbation.view(batch_size, channels, pix, pix)

def proj_l1ball(x, epsilon=10, device = "cuda:0"):
    assert epsilon > 0
#     ipdb.set_trace()
    # compute the vector of absolute values
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        # print (u.sum(dim = (1,2,3)))
         # check if x is already a solution
#         y = x* epsilon/norms_l1(x)
        return x

    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    a = torch.tensor([[
        0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1360, 0.1250, 0.1250, 0.1250,
         0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1193, 0.1167, 0.1167, 0.1167,
         0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111,
         0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769,
         0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0714, 0.0714, 0.0714, 0.0714,
         0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714,
         0.0714, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667,
         0.0667, 0.0667, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
         0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
         0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500,
         0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0000, 0.0000,],
        [0.1714, 0.1667, 0.1667, 0.1667, 0.1611, 0.1484, 0.1333, 0.1333, 0.1250,
         0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1216, 0.1167, 0.1111, 0.1111,
         0.1111, 0.1111, 0.1111, 0.1111, 0.1098, 0.1020, 0.1000, 0.1000, 0.1000,
         0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.0980, 0.0863, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
         0.0833, 0.0833, 0.0824, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769, 0.0769,
         0.0745, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0714, 0.0706,
         0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667, 0.0667,
         0.0667, 0.0611, 0.0588, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
         0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526, 0.0526,
         0.0526, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500,
         0.0500, 0.0500, 0.0471, 0.0471, 0.0431, 0.0392, 0.0392, 0.0314, 0.0314,
         0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0275, 0.0243, 0.0235, 0.0196,
         0.0196, 0.0167, 0.0157, 0.0157, 0.0157, 0.0118, 0.0118, 0.0118, 0.0118,
         0.0078, 0.0078, 0.0064, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,]]).to(device)
    y = my_proj_simplex(u, epsilon=epsilon, device = device)
    # compute the solution to the original problem on v
    y = y.view(-1,3,32,32)
    y *= x.sign()
    return y

def my_proj_simplex(v, epsilon=12, device = "cuda:1"):
    assert epsilon > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    v = v.view(batch_size,-1)
    n = v.shape[1]
    gamma, indices = torch.sort(v, descending = True)
    # print('gamma', gamma)
    torch.set_printoptions(profile="full")
    # print(gamma[:2])
    # input()
    gamma_cumsum = gamma.cumsum(dim = 1)
    js = 1.0 / torch.arange(1, n+1).float().to(device)
    temp = gamma - js * (gamma_cumsum - epsilon)
    rho = (torch.argmin((temp > 0).int().detach().cpu(), dim=1).to(device) - 1) % n
    # print('temp', temp  > 0)
    # print(rho)
    rho_index = torch.stack([torch.arange(batch_size).to(device), rho], dim=0).detach().cpu().numpy()
    eta = (1.0 / (1 + rho.float()) * (gamma_cumsum[rho_index] - epsilon)).unsqueeze(1)
    new_delta = torch.clamp(v - eta, 0)
    # print('check', torch.sum(new_delta, dim=1))
    # input()
    return new_delta

def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex    
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    # print(u)
    # print(u.shape)
    # input('check')
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n+1).to(device).float()
    comp = (vec > (cssv - s))

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

def epoch(loader, lr_schedule,  model, epoch_i, criterion = nn.CrossEntropyLoss(), opt=None, device = "cuda:0", stop = False):
    """Standard training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    loader_bar = tqdm(loader)
    for i,batch in enumerate(loader_bar): 
        X,y = batch['input'], batch['target']
        output = model(X)
        loss = criterion(output, y)        
        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        loader_bar.set_description('Epoch: [{}] Loss:{:.2f} Acc:{:.2f}%'.format(epoch_i, train_loss / train_n, train_acc / train_n * 100))

        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_recon(loader,  model, epoch_i, criterion = nn.CrossEntropyLoss(), opt=None, device = "cuda:0", stop = False):
    """Standard training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    loader_bar = tqdm(loader)
    for i,batch in enumerate(loader_bar): 
        X,y = batch
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)        
        if opt != None:   
            # lr = lr_schedule(epoch_i + (i+1)/len(loader))
            # opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        loader_bar.set_description('Epoch: [{}] Loss:{:.2f} Acc:{:.2f}%'.format(epoch_i, train_loss / train_n, train_acc / train_n * 100))

        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n


def epoch_adversarial_saver(batch_size,loader, model, attack, epsilon, num_iter, device = "cuda:0", restarts = 10):
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0
    # print("Attack: ", attack, " epsilon: ", epsilon )
    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        delta = attack(model, X, y, epsilon = epsilon, num_iter = num_iter, device = device, restarts = restarts)
        output = model(X+delta)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        correct = (output.max(1)[1] == y).float()
        eps = (correct*1000 + epsilon - 0.000001).float()
        train_n += y.size(0)
        break
    return eps,  train_acc / train_n

def epoch_adversarial(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, stats = False, num_stop=500, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    train_l0 = 0
    train_l1 = 0
#     ipdb.set_trace()

    loader_bar = tqdm(loader)
    
    for i,batch in enumerate(loader_bar): 
        # print(i, len(loader))
        X,y = batch['input'], batch['target']
        if stats:
            delta = attack(model, X, y, device = device, batchid = i, **kwargs)
        else:
            delta = attack(model, X, y, device = device, **kwargs)

        batch_l0 = torch.sum(norms_l0(delta)).item()
        batch_l1 = torch.sum(norms_l1(delta)).item()
        train_l0 += batch_l0
        train_l1 += batch_l1
        
        output = model(X+delta)
        # imshow(X[11])
        # print (X[11])
        # imshow((X+delta)[11])
        # print (norms_l1(delta))
#         output = model(X)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        loader_bar.set_description('Epoch: [{}] Loss:{:.2f} Acc:{:.2f}% Norm_l0:{:.2f} Norm_l1:{:.2f}'.format(epoch_i, train_loss / train_n, train_acc / train_n * 100, train_l0 / train_n, train_l1 / train_n))

        if opt != None:   
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            if (stop):
                if train_n >= num_stop:
                    break
        
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial_recon(loader, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
    opt=None, device = "cuda:0", stop = False, stats = False, return_feature=False, num_stop=500, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    train_loss = 0
    train_acc = 0
    train_n = 0
    train_l0 = 0
    train_l1 = 0
    feature_list = []
#     ipdb.set_trace()

    loader_bar = tqdm(loader)
    
    for i,batch in enumerate(loader_bar): 
        # print(i, len(loader))
        X,y = batch
        X, y = X.to(device), y.to(device)
        if stats:
            delta = attack(model, X, y, device = device, batchid = i, **kwargs)
        else:
            delta = attack(model, X, y, device = device, **kwargs)

        batch_l0 = torch.sum(norms_l0(delta)).item()
        batch_l1 = torch.sum(norms_l1(delta)).item()
        train_l0 += batch_l0
        train_l1 += batch_l1
        
        if return_feature:
            feature, output = model(X+delta, output_feature = True)
            feature_list.append(feature.detach())
        else:
            output = model(X+delta, output_feature = False)


        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        loader_bar.set_description('Epoch: [{}] Loss:{:.2f} Acc:{:.2f}% Norm_l0:{:.2f} Norm_l1:{:.2f}'.format(epoch_i, train_loss / train_n, train_acc / train_n * 100, train_l0 / train_n, train_l1 / train_n))

        if opt != None:   
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            if (stop):
                if train_n >= num_stop:
                    break
    
    if return_feature:
        return train_loss / train_n, train_acc / train_n, torch.cat(feature_list, dim=0)
    else:
        return train_loss / train_n, train_acc / train_n

def triple_adv(loader, lr_schedule, model, epoch_i, attack,  criterion = nn.CrossEntropyLoss(),
                     opt=None, device= "cuda:0", epsilon_l_1 = 12, epsilon_l_2 = 0.5, epsilon_l_inf = 0.03, num_iter = 50):
    
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i,batch in enumerate(loader): 
        X,y = batch['input'], batch['target']
        lr = lr_schedule(epoch_i + (i+1)/len(loader))
        opt.param_groups[0].update(lr=lr)
        ##Always calls the default version 0 for the individual attacks

        #L1
        delta = pgd_l1_topk(model, X, y, device = device, epsilon = epsilon_l_1)
        output = model(X+delta)
        loss = criterion(output,y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        #L2
        delta = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2)
        output = model(X+delta)
        loss = nn.CrossEntropyLoss()(output,y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        

        #Linf
        delta = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf)
        output = model(X+delta)
        loss = nn.CrossEntropyLoss()(output,y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        else:
            break
        # break
    return train_loss / train_n, train_acc / train_n

def triple_adv_recon(loader, model, epoch_i, attack,  criterion = nn.CrossEntropyLoss(), opt=None, device= "cuda:0", epsilon_l_1 = 12, alpha_l1 = 1, epsilon_l_2 = 0.5, alpha_l2 = 0.166666, epsilon_l_inf_255 = 8, alpha_linf_255 = 2, num_iter = 10, restarts=0):
    
    train_loss = 0
    train_acc = 0
    train_n = 0

    loader_bar = tqdm(loader)

    for i,batch in enumerate(loader_bar): 
        X,y = batch
        X, y = X.to(device), y.to(device)
        # lr = lr_schedule(epoch_i + (i+1)/len(loader))
        # opt.param_groups[0].update(lr=lr)
        ##Always calls the default version 0 for the individual attacks
        loss = 0

        l1_start = time.time()

        #L1
        # time0 = time.time()
        delta_l1 = pgd_l1_sign_free(model, X, y, device = device, epsilon = epsilon_l_1, alpha = alpha_l1, restarts=restarts, num_iter=num_iter)
        # time1 = time.time()
        output_l1 = model(X+delta_l1)
        # time2 = time.time()
        loss_l1 = nn.CrossEntropyLoss()(output_l1,y)
        # time3 = time.time()
        loss += loss_l1 / 3

        train_loss += loss_l1.item()*y.size(0)
        train_acc += (output_l1.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        l2_start = time.time()

        # print('time0:', time1 - time0, 'time1:', time2 - time1, 'time2:', time3 - time2)
        
        #L2
        delta_l2 = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2, alpha = alpha_l2, restarts=restarts, num_iter=num_iter)
        output_l2 = model(X+delta_l2)
        loss_l2 = nn.CrossEntropyLoss()(output_l2,y)
        loss += loss_l2 / 3

        train_loss += loss_l2.item()*y.size(0)
        train_acc += (output_l2.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        linf_start = time.time()

        #Linf
        delta_linf = pgd_linf(model, X, y, device = device, epsilon_255 = epsilon_l_inf_255, alpha_255 = alpha_linf_255, restarts=restarts, num_iter=num_iter)
        output_linf = model(X+delta_linf)
        loss_linf = nn.CrossEntropyLoss()(output_linf,y)
        loss += loss_linf / 3

        train_loss += loss_linf.item()*y.size(0)
        train_acc += (output_linf.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        bp_start = time.time()

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        else:
            break
        # break

        bp_end = time.time()

        loader_bar.set_description('Epoch: [{}] Loss:{:.2f} Acc:{:.2f}%'.format(epoch_i, train_loss / train_n, train_acc / train_n * 100))


    return train_loss / train_n, train_acc / train_n


def test_visualization(net, data_loader, save_name, model2=None):
    net.eval()
    c = 10
    feature_bank = []
    feature_l1_bank = []
    feature_l2_bank = []
    feature_linf_bank = []
    labels_bank = []
    count_data = 0
    # feature_bank2 = []
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    for data, labels in data_loader:
        data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        index = labels <= 3
        data = data[index]
        labels = labels[index]
        delta_1 = pgd_l1_sign_free(net, data, labels, device = labels.device, epsilon = 12, alpha = 1, restarts=0, num_iter=10)
        delta_2 = pgd_l2(net, data, labels, device = labels.device, epsilon = 0.5, alpha = 0.16666, restarts=0, num_iter=10)
        delta_inf = pgd_linf(net, data, labels, device = labels.device, epsilon_255 = 8, alpha_255 = 0.8, restarts=0, num_iter=10)

        feature_l1_bank.append(delta_1)
        feature_l2_bank.append(delta_2)
        feature_linf_bank.append(delta_inf)

        feature_bank.append(data)
        labels_bank.append(labels)
        count_data += data.shape[0]
        if count_data >= 500:
            break
    feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    sample_num = feature_bank.shape[0]
    feature_l1_bank = torch.cat(feature_l1_bank, dim=0).contiguous()
    feature_l2_bank = torch.cat(feature_l2_bank, dim=0).contiguous()
    feature_linf_bank = torch.cat(feature_linf_bank, dim=0).contiguous()
    feature_labels = torch.cat(labels_bank, dim=0).contiguous()

    with torch.no_grad():
        feature, out = net(feature_bank.cuda(non_blocking=True), output_feature=True)
        feature_l1, out = net(feature_l1_bank.cuda(non_blocking=True), output_feature=True)
        feature_l2, out = net(feature_l2_bank.cuda(non_blocking=True), output_feature=True)
        feature_linf, out = net(feature_linf_bank.cuda(non_blocking=True), output_feature=True)
        # if model2 != None:
        #     feature_bank2 = torch.cat(feature_bank2, dim=0).contiguous()
        #     feature_bank = torch.cat([feature_bank, feature_bank2], dim=0).contiguous()
        # [N]
        feature = torch.cat([feature, feature_l1, feature_l2, feature_linf], dim=0)
        # feature_labels = torch.cat([feature_labels, feature_labels, feature_labels, feature_labels], dim=0)
        print(feature.shape)
        feature_tsne_input = feature.cpu().numpy()
        labels_tsne_color = feature_labels.cpu().numpy()
        # if model2 != None:
        #     labels_tsne_color = np.concatenate([labels_tsne_color, labels_tsne_color], axis=0)
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("Feature space")
        cm = plt.cm.get_cmap('gist_rainbow', c)
        plt.scatter(feature_tsne_output[:sample_num, 0], feature_tsne_output[:sample_num, 1], s=15, c=labels_tsne_color, cmap=cm)
        plt.scatter(feature_tsne_output[sample_num:2*sample_num, 0], feature_tsne_output[sample_num:2*sample_num, 1], s=30, c=labels_tsne_color, cmap=cm, marker='x')
        plt.scatter(feature_tsne_output[2*sample_num:3*sample_num, 0], feature_tsne_output[2*sample_num:3*sample_num, 1], s=20, c=labels_tsne_color, cmap=cm, marker='s')
        plt.scatter(feature_tsne_output[3*sample_num:, 0], feature_tsne_output[3*sample_num:, 1], s=30, c=labels_tsne_color, cmap=cm, marker='^')
        # plt.scatter(feature_tsne_output[1024:, 0], feature_tsne_output[1024:, 1], s=10, c=labels_tsne_color[1024:], cmap=cm, marker='+')
        ax.xaxis.set_major_formatter(NullFormatter())  # ??????????????????????????????
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./visualization/feature_{}.png'.format(save_name))

    return 

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def trades(loader,
            model,
            optimizer,
            epoch_i, 
            epsilon_l1=12,
            alpha_l1=1,
            step_size_255=0.8,
            epsilon_255=8,
            perturb_steps=10,
            beta=1.0,
            mu=0.5,
            distance='l_inf',
            device= "cuda:0",
            feature_space=False,
            ):
    epsilon = float(epsilon_255) / 255.
    step_size = float(step_size_255) / 255.

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)

    train_loss = 0
    train_acc = 0
    train_n = 0

    loader_bar = tqdm(loader)

    for i,batch in enumerate(loader_bar): 

        x_natural, y = batch
        x_natural, y = x_natural.to(device), y.to(device)

        model.eval()

        batch_size = len(x_natural)

        # generate adversarial example
        if distance in ['linf', 'l1_linf'] and mu != 1:
            x_adv_linf = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            for _ in range(perturb_steps):
                x_adv_linf.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv_linf), dim=1), F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv_linf])[0]
                x_adv_linf = x_adv_linf.detach() + step_size * torch.sign(grad.detach())
                x_adv_linf = torch.min(torch.max(x_adv_linf, x_natural - epsilon), x_natural + epsilon)
                x_adv_linf = torch.clamp(x_adv_linf, 0.0, 1.0)

        if distance in ['l2']:
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                            F.softmax(model(x_natural), dim=1))
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)

        if distance in ['l1', 'l1_linf'] and mu != 0:
            x_adv_l1 = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
            for _ in range(perturb_steps):
                x_adv_l1.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv_l1), dim=1), F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv_l1])[0]
                delta = grad.detach()
                grad_norm = norms(delta.detach())

                if (grad_norm == 0).any():
                    # print(delta.detach()[0])
                    # print(norms(delta.detach()).squeeze(1).squeeze(1).squeeze(1))
                    # print((norms(delta.detach()) == 0).squeeze(1).squeeze(1).squeeze(1))
                    # raise('Norm is 0 again')
                    grad_norm[grad_norm == 0] = 1

                delta = x_adv_l1.detach() - x_natural.detach() + alpha_l1 * delta.detach() / grad_norm.view(-1, 1, 1, 1)

                if (norms_l1(delta) > epsilon_l1).any():
                    # input('check proj_l1ball in pgd_l1_topk')
                    delta.data = proj_l1ball(delta.data, epsilon_l1, device)
                    # print(norms_l1(delta) > epsilon)

                x_adv_l1 = x_natural + delta
                x_adv_l1 = torch.clamp(x_adv_l1, 0.0, 1.0)

        # else:
        #     x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()

        if distance == 'l1' or (distance == 'l1_linf' and mu != 0):
            x_adv_l1 = Variable(torch.clamp(x_adv_l1, 0.0, 1.0), requires_grad=False)
        if distance == 'linf' or (distance == 'l1_linf' and mu != 1):
            x_adv_linf = Variable(torch.clamp(x_adv_linf, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        if distance in ['l1']:
            if not feature_space:
                l1_distrib = model(x_adv_l1)
                natural_distrib = model(x_natural)
            else:
                l1_distrib, _ = model(x_adv_l1, output_feature=True)
                natural_distrib, _ = model(x_natural, output_feature=True)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(l1_distrib, dim=1),
                                                            F.softmax(natural_distrib, dim=1))
        elif distance in ['linf']:
            if not feature_space:
                linf_distrib = model(x_adv_linf)
                natural_distrib = model(x_natural)
            else:
                linf_distrib, _ = model(x_adv_linf, output_feature=True)
                natural_distrib, _ = model(x_natural, output_feature=True)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(linf_distrib, dim=1),
                                                            F.softmax(natural_distrib, dim=1))
        elif distance in ['l1_linf']:
            loss_robust = 0
            if mu != 0:
                loss_robust += mu * (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv_l1), dim=1),
                                                                F.softmax(model(x_natural), dim=1))
            if mu != 1:
                loss_robust += (1-mu) * (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv_linf), dim=1),
                                                                   F.softmax(model(x_natural), dim=1))
            if mu != 0 and mu != 1:
                loss_robust += 0.5 * (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv_l1), dim=1),
                                                                   F.softmax(model(x_adv_linf), dim=1))
                loss_robust += 0.5 * (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv_linf), dim=1),
                                                                   F.softmax(model(x_adv_l1), dim=1))
                
        
        loss = loss_natural + beta * loss_robust

        # print(loss_robust.item()*y.size(0))

        train_loss += loss.item()*y.size(0)
        train_acc += (logits.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        else:
            break
        # break

        loader_bar.set_description('Epoch: [{}] Loss:{:.2f} Acc:{:.2f}%'.format(epoch_i, train_loss / train_n, train_acc / train_n * 100))


    return train_loss / train_n, train_acc / train_n
