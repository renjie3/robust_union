import argparse

# python3 train.py -gpu_id 0 -model 3 -batch_size 128 -lr_schedule 1
parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("--model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=str, default = 'vanilla')
parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 128)", type = int, default = 128)
parser.add_argument("--dataset", type=str, default = 'cifar10')
parser.add_argument("--epsilon_l_1", type=int, default = 12)
parser.add_argument("--alpha", type=float, default = 1)
parser.add_argument("--epsilon_l_2", type=float, default = 0.5)
parser.add_argument("--alpha_l2", type=float, default = 0.166666)
parser.add_argument("--epsilon_l_inf_255", type=int, default = 8)
parser.add_argument("--alpha_linf_255", type=float, default = 2)
parser.add_argument("--num_iter", type=int, default = 50)
parser.add_argument("--num_stop", type=int, default = 500)
parser.add_argument("--epochs", type=int, default = 50)
parser.add_argument("--restarts", type=int, default = 0)
parser.add_argument("--sign_free", action='store_true', default=False)
parser.add_argument("--alpha_sign_free", type=float, default = 0.5)
parser.add_argument("--trades_distance", type=str, default = 'l_inf')
parser.add_argument("--beta", type=float, default = 1.0)
parser.add_argument("--mu", type=float, default = 0.5)
parser.add_argument("--feature_space", action='store_true', default=False)
parser.add_argument("--seed", type=int, default = 0)
parser.add_argument("--lr", type=float, default = 0.05)
parser.add_argument("--load_model", action='store_true', default=False)
parser.add_argument("--load_model_path", type=str, default = '')
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--job_id", type=str, default = 'local')
parser.add_argument('--local', default='', type=str, help='The gpu number used on developing node.')

params = parser.parse_args()

import os
if params.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = params.local

import sys
sys.path.append('./models/')
import torch
from models import PreActResNet18, ResNet18
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
# from torch.optim.lr_scheduler import SequentialLR
sys.path.append('./utils/')
from core import *
from torch_backend import *
from cifar_funcs import *
import ipdb
import time

device_id = params.gpu_id

device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))

if torch.cuda.is_available():
    # torch.manual_seed(args.seed)
    torch.cuda.manual_seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')

torch.cuda.device_count() 
batch_size = params.batch_size
choice = params.model

epochs = params.epochs
DATA_DIR = './data'
# dataset = cifar10(DATA_DIR)

train_set, test_set = set_dataset(DATA_DIR, params)

# train_set = list(zip(transpose(normalise2(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
# test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
# train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])
# train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2, gpu_id = torch.cuda.current_device())
# test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

if params.dataset == 'cifar10':
    num_classes = 10
elif params.dataset == 'cifar100':
    num_classes = 100

# model = PreActResNet18(num_classes=num_classes).cuda()
model = ResNet18().to(device)
# for m in model.children(): 
#     if not isinstance(m, nn.BatchNorm2d):
#         m.half()   

if params.load_model:
    model.load_state_dict(torch.load("{}.pt".format(params.load_model_path), map_location = device))
        
opt = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# lr_schedule_func = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0.05, 0.1, 0.05, 0])[0]

# lr_schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_schedule_func)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs - 10], gamma=0.1)
#For clearing pytorch cuda inconsistency
try:
    train_loss, train_acc = epoch(test_batches, lr_schedule, model, 0, criterion, opt = None, device = device, stop = True)
except:
    a =1

# attack_name = [0"pgd_linf", 1"pgd_l1_topk", 2"pgd_l2", 3"msd_v0", 4"triple_adv", 5"pgd_worst_dir", 6"vanilla", 7"pgd_l1_sign_free"]

attack_list = {"pgd_linf": pgd_linf,  "pgd_l1_topk": pgd_l1_topk, "pgd_l2": pgd_l2, "msd_v0": msd_v0, "triple_adv": triple_adv, "pgd_worst_dir": pgd_worst_dir, "vanilla": triple_adv, "pgd_l1_sign_free": pgd_l1_sign_free, "trades": trades}
# attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "msd_v0", "triple_adv", "pgd_worst_dir", "vanilla", "pgd_l1_sign_free"]
# folder_name = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA", "pgd_l1_sign_free"]

model_dir = "Final/{0}_{1}".format(choice, params.job_id)

if(not os.path.exists(model_dir)):
    os.makedirs(model_dir)

file = open("{0}/logs.txt".format(model_dir), "w")

def myprint(a):
    print(a)
    file.write(a)
    file.write("\n")

attack = attack_list[choice]
print(choice)

for epoch_i in range(1,epochs+1):
    start_time = time.time()
    # lr = lr_schedule_func(epoch_i + (epoch_i+1)/len(train_loader))
    # print(lr)
    if choice == "vanilla":
        train_loss, train_acc = epoch_recon(train_loader, model, epoch_i, criterion, opt = opt, device = device)
    elif choice == "triple_adv":
        train_loss, train_acc = triple_adv_recon(train_loader, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_1 = params.epsilon_l_1, alpha_l1 = params.alpha, epsilon_l_2 = params.epsilon_l_2, alpha_l2 = params.alpha_l2, epsilon_l_inf_255 = params.epsilon_l_inf_255, alpha_linf_255 = params.alpha_linf_255, num_iter = params.num_iter, restarts=params.restarts)
    elif choice == "msd_v0":
        train_loss, train_acc = epoch_adversarial_recon(train_loader, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_1 = params.epsilon_l_1, alpha_l1 = params.alpha, epsilon_l_2 = params.epsilon_l_2, alpha_l2 = params.alpha_l2, epsilon_l_inf_255 = params.epsilon_l_inf_255, alpha_linf_255 = params.alpha_linf_255, num_iter = params.num_iter)
    elif choice == "pgd_worst_dir":
        train_loss, train_acc = epoch_adversarial_recon(train_loader, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_l_1 = params.epsilon_l_1, alpha_l1 = params.alpha, epsilon_l_2 = params.epsilon_l_2, alpha_l2 = params.alpha_l2, epsilon_l_inf_255 = params.epsilon_l_inf_255, alpha_linf_255 = params.alpha_linf_255, num_iter = params.num_iter, restarts=params.restarts)
    elif choice == "pgd_l1_sign_free":
        train_loss, train_acc = epoch_adversarial_recon(train_loader, model, epoch_i, attack, criterion, opt = opt, device = device, alpha = params.alpha, num_iter = params.num_iter, restarts=params.restarts)
    elif choice == "pgd_linf":
        train_loss, train_acc = epoch_adversarial_recon(train_loader, model, epoch_i, attack, criterion, opt = opt, device = device, epsilon_255 = params.epsilon_l_inf_255, alpha_255 = params.alpha_linf_255, num_iter = params.num_iter, restarts=params.restarts, sign_free=params.sign_free, alpha_sign_free=params.alpha_sign_free)
    elif choice == "trades":
        train_loss, train_acc = trades(train_loader, model, opt, epoch_i, epsilon_l1 = params.epsilon_l_1, alpha_l1 = params.alpha, step_size_255=params.alpha_linf_255, epsilon_255=params.epsilon_l_inf_255, perturb_steps=params.num_iter, beta=params.beta, mu=params.mu, distance=params.trades_distance, device = device, feature_space=params.feature_space)
    else:
        pass
        # train_loss, train_acc = epoch_adversarial_recon(train_loader, model, epoch_i, attack, criterion, opt = opt, device = device, alpha = params.alpha, num_iter = params.num_iter, restarts=params.restarts)
    lr_schedule.step()

    # model.eval()

    if epoch_i % 5 == 0 or epoch_i < 5:
        total_loss, total_acc = epoch_recon(test_loader, model, epoch_i, criterion, opt = None, device = device)
        total_loss, total_acc_1_sign_free = epoch_adversarial_recon(test_loader, model, epoch_i,  pgd_l1_sign_free, criterion, opt = None, device = device, stop = True, num_stop=params.num_stop, alpha=params.alpha, num_iter=params.num_iter)
        total_loss, total_acc_1 = epoch_adversarial_recon(test_loader, model, epoch_i,  pgd_l1_topk, criterion, opt = None, device = device, stop = True, num_stop=params.num_stop)
        total_loss, total_acc_2 = epoch_adversarial_recon(test_loader, model, epoch_i,  pgd_l2, criterion, opt = None, device = device, stop = True, num_stop=params.num_stop,)
        total_loss, total_acc_3 = epoch_adversarial_recon(test_loader, model, epoch_i,  pgd_linf, criterion, opt = None, device = device, stop = True, num_stop=params.num_stop,)
        if params.debug and params.load_model:
            sys.exit()
        myprint('Epoch: {7}, Clean Acc: {6:.4f} Train Acc: {5:.4f}, Test Acc 1_sign_free: {4:.4f}, Test Acc 1: {3:.4f}, Test Acc 2: {2:.4f}, Test Acc inf: {1:.4f}, Time: {0:.1f}'.format(time.time()-start_time, total_acc_3, total_acc_2,total_acc_1, total_acc_1_sign_free, train_acc, total_acc, epoch_i))    
        if epoch_i >= 40:
            torch.save(model.state_dict(), "{0}/iter_{1}.pt".format(model_dir, str(epoch_i)))
        if params.debug:
            torch.save(model.state_dict(), "{0}/iter_{1}.pt".format(model_dir, str(epoch_i)))
            sys.exit()
