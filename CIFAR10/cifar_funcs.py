import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import ipdb
import random
from tqdm import tqdm

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


def pgd_l1_topk(model, X,y, epsilon = 12, alpha = 1, num_iter = 20, k = 20, device = "cuda:1", restarts = 1, version = 0):
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
        delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X, k)
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
            k = random.randint(80,99)
            # alpha = 0.05/k*20
            delta.data += alpha*correct*l1_dir_topk(delta.grad.detach(), delta.data, X,k)
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

def pgd_linf(model, X, y, epsilon=0.03, alpha=0.003, num_iter = 10, device = "cuda:0", restarts = 0, version = 0):
    epsilon = 8. / 255.
    alpha = 1. / 255.
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
        delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
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
            delta.data = (delta.data + alpha*correct*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
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

def msd_v0(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
                alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 0.05, num_iter = 50, device = "cuda:0"):
    delta = torch.zeros_like(X,requires_grad = True)
    max_delta = torch.zeros_like(X)
    max_max_delta = torch.zeros_like(X)
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_max_loss = torch.zeros(y.shape[0]).to(y.device)
    alpha_l_1_default = alpha_l_1
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        with torch.no_grad():                
            #For L_2
            delta_l_2  = delta.data + alpha_l_2*delta.grad / norms(delta.grad)      
            delta_l_2 *= epsilon_l_2 / norms(delta_l_2).clamp(min=epsilon_l_2)
            delta_l_2  = torch.min(torch.max(delta_l_2, -X), 1-X) # clip X+delta to [0,1]

            #For L_inf
            delta_l_inf=  (delta.data + alpha_l_inf*delta.grad.sign()).clamp(-epsilon_l_inf,epsilon_l_inf)
            delta_l_inf = torch.min(torch.max(delta_l_inf, -X), 1-X) # clip X+delta to [0,1]

            #For L1
            k = random.randint(5,20)
            alpha_l_1 = (alpha_l_1_default/k)*20
            delta_l_1  = delta.data + alpha_l_1*l1_dir_topk(delta.grad, delta.data, X, alpha_l_1, k = k)
            delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
            delta_l_1  = torch.min(torch.max(delta_l_1, -X), 1-X) # clip X+delta to [0,1]
            
            #Compare
            delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
            max_loss = torch.zeros(y.shape[0]).to(y.device)
            for delta_temp in delta_tup:
                loss_temp = nn.CrossEntropyLoss(reduction = 'none')(model(X + delta_temp), y)
                max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                max_loss = torch.max(max_loss, loss_temp)
            delta.data = max_delta.data
            max_max_delta[max_loss> max_max_loss] = max_delta[max_loss> max_max_loss]
            max_max_loss[max_loss> max_max_loss] = max_loss[max_loss> max_max_loss]
        delta.grad.zero_()

    return max_max_delta


def pgd_worst_dir(model, X,y, epsilon_l_inf = 0.03, epsilon_l_2= 0.5, epsilon_l_1 = 12, 
    alpha_l_inf = 0.003, alpha_l_2 = 0.05, alpha_l_1 = 0.05, num_iter = 100, device = "cuda:0"):
    #Always call version = 0
    delta_1 = pgd_l1_topk(model, X, y, epsilon = epsilon_l_1, alpha = alpha_l_1,  device = device)
    delta_2 = pgd_l2(model, X, y, epsilon = epsilon_l_2, alpha = alpha_l_2,  device = device)
    delta_inf = pgd_linf(model, X, y, epsilon = epsilon_l_inf, alpha = alpha_l_inf, device = device)
    
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

def l1_dir_topk(grad, delta, X, k=20):
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
    num_ties = np.sum(tied_for_max, (1, 2, 3), keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties

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
        
        output = model(X+delta)

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

def triple_adv(loader, model, epoch_i, attack,  criterion = nn.CrossEntropyLoss(),
                     opt=None, device= "cuda:0", epsilon_l_1 = 12, epsilon_l_2 = 0.5, epsilon_l_inf = 0.03, num_iter = 50):
    
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
