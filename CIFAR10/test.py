import argparse

parser = argparse.ArgumentParser(description='Adversarial Training for CIFAR10', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-gpu_id", help="Id of GPU to be used", type=int, default = 0)
parser.add_argument("-model", help="Type of Adversarial Training: \n\t 0: l_inf \n\t 1: l_1 \n\t 2: l_2 \n\t 3: msd \n\t 4: triple \n\t 5: worst \n\t 6: vanilla", type=int, default = 3)
parser.add_argument("-batch_size", help = "Batch Size for Test Set (Default = 1000)", type = int, default = 1000)
parser.add_argument("-attack", help = "Foolbox = 0; Custom PGD = 1, Min PGD = 2, Fast DDN = 3", type = int, default = 0)
parser.add_argument("-restarts", help = "Default = 10", type = int, default = 2)
parser.add_argument("-path", help = "To override default model fetching- Automatically appends '.pt' to path", type = str)
parser.add_argument("-subset", help = "Subset of attacks", type = int, default = -1)
parser.add_argument("--dataset", type=str, default = 'cifar10')
parser.add_argument("--alpha", type=float, default = 0.05)
parser.add_argument("--alpha_topk", type=float, default = 0.05)
parser.add_argument("--alpha_top1", type=float, default = 0.05)
parser.add_argument("--pgd_norm", type=int, default = 0)
parser.add_argument("--num_iter", type=int, default = 50)
parser.add_argument("--num_stop", type=int, default = 500)
parser.add_argument("--seed", type=int, default = 0)
parser.add_argument('--test_subset', action='store_true', default=False)
parser.add_argument("--job_id", type=str, default = 'local')
parser.add_argument('--local', default='', type=str, help='The gpu number used on developing node.')


params = parser.parse_args()

import os
if params.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = params.local

import sys
sys.path.append('./models/')
import torch
from models import PreActResNet18
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
sys.path.append('./utils/')
from core import *
from torch_backend import *
import ipdb 
import foolbox
import foolbox.attacks as fa
from cifar_funcs import *
from time import time
from fast_adv.attacks import DDN

# python3 test.py -gpu_id 0 -model 0 -batch_size 1 -attack 0 -restarts 10

device_id = params.gpu_id
batch_size = params.batch_size
choice = params.model
attack = params.attack
res = params.restarts
path = params.path
subset = params.subset


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

DATA_DIR = './data'
dataset = cifar10(DATA_DIR)
t = Timer()

print('Preprocessing test data')
test_set = list(zip(transpose(normalise2(dataset['test']['data'])), dataset['test']['labels']))
print('Finished in {0:.2} seconds'.format(t()))


def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'SAPA':
        metric = foolbox.distances.L0
        A = fa.SaltAndPepperNoiseAttack(fmodel, distance = metric)
    elif attack == 'PA':
        metric = foolbox.distances.L0
        A = fa.PointwiseAttack(fmodel, distance = metric)

    # L2
    elif 'IGD' in attack:
        metric = foolbox.distances.MSE
        A = fa.L2BasicIterativeAttack(fmodel, distance = metric)
        # kwargs['epsilons'] = 1.5
    elif attack == 'AGNA':
        metric = foolbox.distances.MSE
        kwargs['epsilons'] = np.linspace(0.5, 1, 50)
        A = fa.AdditiveGaussianNoiseAttack(fmodel, distance = metric)
    elif attack == 'BA':
        metric = foolbox.distances.MSE
        A = fa.BoundaryAttack(fmodel, distance = metric)
        kwargs['log_every_n_steps'] = 500001
    elif 'DeepFool' in attack:
        metric = foolbox.distances.MSE
        A = fa.DeepFoolL2Attack(fmodel, distance = metric)
    elif attack == 'PAL2':
        metric = foolbox.distances.MSE
        A = fa.PointwiseAttack(fmodel, distance = metric)
    elif attack == "CWL2":
        metric = foolbox.distances.MSE
        A = fa.CarliniWagnerL2Attack(fmodel, distance = metric)

    # L inf
    elif 'FGSM' in attack and not 'IFGSM' in attack:
        metric = foolbox.distances.Linf
        A = fa.FGSM(fmodel, distance = metric)
        kwargs['epsilons'] = 20
    elif 'PGD' in attack:
        metric = foolbox.distances.Linf
        A = fa.LinfinityBasicIterativeAttack(fmodel, distance = metric)
    elif 'IGM' in attack:
        metric = foolbox.distances.Linf
        A = fa.MomentumIterativeAttack(fmodel, distance = metric)
    else:
        raise Exception('Not implemented')
    return A, metric, args, kwargs


def test_foolbox(model_name, max_tests):
    #Saves the minimum epsilon value for successfully attacking each image via different foolbox attacks as an npy file in the folder corresponding to model_name
    #No Restarts in case of BA
    #Batch size = 1 is supported

    print(model_name)
    torch.manual_seed(0)
    model = PreActResNet18().cuda()
    # for m in model.children(): 
    #     if not isinstance(m, nn.BatchNorm2d):
    #         m.half()   
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    model.eval()    
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.float() 
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, device = device)

    if subset == 0:
        attacks_list = ['PA','SAPA']
        types_list   = [ 0    , 0 ]
    elif subset == 1:
        types_list   = [ 2  ]
        attacks_list = ['BA']
    elif subset == 2:
        attacks_list = ['IGD','AGNA','DeepFool','PAL2']
        types_list = [2,2,2,2]
    elif subset == 3 :
        attacks_list =['FGSM','PGD','IGM']
        types_list = [3,3,3]
    elif subset == 4 :
        types_list   = [ 2  ]
        attacks_list = ['CWL2']
    else:
        attacks_list = ['SAPA','PA','IGD','AGNA','BA','DeepFool','PAL2','CWL2''FGSM','PGD','IGM']
        types_list   = [ 0    , 0  , 2   , 2    ,  2  ,  2  ,      2    , 2,   3      , 3   , 3 ]

    norm_dict = {0:norms_l0, 1:norms_l1, 2:norms,3:norms_linf}
    
    for j in range(len(attacks_list)):
        restarts = res
        attack_name = attacks_list[j]
        print (attack_name)
        types = types_list[j]
        norm = norm_dict[types]
        max_check = max_tests
        test_batches = Batches(test_set, batch_size = 1, shuffle=False, gpu_id = torch.cuda.current_device())

        if attack_name == "BA":
            restarts = 1

        output = np.ones((max_check))
        start = time()

        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        err = 0
        for i,batch in enumerate(test_batches): 
            start = time()
            distance = 1000
            X,y = batch['input'].float(), batch['target'].float()
            image = X[0,:,:,:].view(3,32,32).detach().cpu().numpy().astype('float32')
            label = y[0].long().item()
            for r in range (restarts):
                try:
                    adversarial = attack(image, label=label) if (attack_name !='CWL2') else attack(image, label=label, max_iterations = 200,  learning_rate=0.025)
                except:
                    if (i == 0):
                        adversarial = attack(image, label=label) if (attack_name !='CWL2') else attack(image, label=label, max_iterations = 200,  learning_rate=0.025)
                    else:
                        print ("assertion error")
                        continue
                try :
                    adversarial.all()
                    adv = torch.from_numpy(adversarial).float().view(1,3,32,32).to(device)
                    distance = min(distance, norm(X - adv).item())
                except:
                    a = 0
                    continue

            output[total] = distance
            total += 1
            print(total, " ", attack_name, " " ,model_name, " Time taken = ", time() - start, " distance = ", str(distance))
            if (total >= max_check):
                np.save(model_name + "/" + attack_name + ".npy" ,output)
                break

        print("Time Taken = ", time() - start)


def test_pgd(model_name, clean = False):
    #Computes the adversarial accuracy at standard thresholds of (0.3,1.5,12) for first 1000 images

    print(model_name)
    print(device)
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())
    if params.dataset == 'cifar10':
        num_classes = 10
    elif params.dataset == 'cifar100':
        num_classes = 100
    model = PreActResNet18(num_classes=num_classes).to(device)
    # for m in model.children(): 
    #     if not isinstance(m, nn.BatchNorm2d):
    #         m.half()   
            
    criterion = nn.CrossEntropyLoss()

    # import time
    start_time = time()

    model.load_state_dict(torch.load(model_name+".pt", map_location = device))
    model.eval()
    lr_schedule = None
    epoch_i = 0
    
    try:
        total_loss, total_acc = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = device, stop = False)
    except:
        raise('problem')
        total_loss, total_acc = epoch(test_batches, lr_schedule, model, epoch_i, criterion, opt = None, device = device, stop = False)
    print('Test Acc Clean: {0:.4f}'.format(total_acc))
    if params.pgd_norm == 7:
        total_loss, total_acc_l1_sign_free = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l1_sign_free, device = device, stop = params.test_subset, num_stop=params.num_stop, restarts = res, alpha = params.alpha, num_iter = params.num_iter)
        print('Test Acc L1 signfree: {0:.4f}'.format(total_acc_l1_sign_free))
    elif params.pgd_norm == 10:
        total_loss, total_acc_l1_sign_free = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l1_sign_free_momentum, device = device, stop = params.test_subset, num_stop=params.num_stop, restarts = res, alpha = params.alpha, num_iter = params.num_iter)
        print('Test acc_l1_sign_free_momentum: {0:.4f}'.format(total_acc_l1_sign_free))
    elif params.pgd_norm == 1:
        total_loss, total_acc_1 = epoch_adversarial(test_batches, None,  model, epoch_i, pgd_l1_topk, device = device, stop = True, num_stop=params.num_stop, restarts = res, num_iter = params.num_iter, alpha=params.alpha)
        print('Test Acc 1: {0:.4f}'.format(total_acc_1))
    elif params.pgd_norm == 2:
        total_loss, total_acc_2 = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l2, device = device, stop = True, num_stop=params.num_stop, restarts = res, epsilon = 0.5, num_iter = params.num_iter, alpha = params.alpha_l2)
        print('Test Acc 2: {0:.4f}'.format(total_acc_2))
    elif params.pgd_norm == 0:
        total_loss, total_acc_inf = epoch_adversarial(test_batches, None, model, epoch_i, pgd_linf, device = device, stop = True, num_stop=params.num_stop, num_iter = params.num_iter, restarts = res)
        print('Test Acc Inf: {0:.4f}'.format(total_acc_inf))
    elif params.pgd_norm == 8:
        total_loss, total_acc_1 = epoch_adversarial(test_batches, None,  model, epoch_i, pgd_l1_top1, device = device, stop = True, num_stop=params.num_stop, restarts = res, num_iter = params.num_iter, alpha=params.alpha)
        print('Test Acc 1: {0:.4f}'.format(total_acc_1))
    elif params.pgd_norm == 9:
        total_loss, total_acc_l1_sign_free = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l1_sign_free, device = device, stop = params.test_subset, num_stop=params.num_stop, restarts = res, alpha = params.alpha, num_iter = params.num_iter)
        print('Test Acc L1 signfree: {0:.4f}'.format(total_acc_l1_sign_free))
        total_loss, total_acc_1 = epoch_adversarial(test_batches, None,  model, epoch_i, pgd_l1_topk, device = device, stop = True, num_stop=params.num_stop, restarts = res, num_iter = params.num_iter, alpha=params.alpha_topk)
        print('Test Acc 1: {0:.4f}'.format(total_acc_1))
        # total_loss, total_acc_1 = epoch_adversarial(test_batches, None,  model, epoch_i, pgd_l1_top1, device = device, stop = True, num_stop=params.num_stop, restarts = res, num_iter = params.num_iter, alpha=params.alpha_top1)
        # print('Test Acc 1: {0:.4f}'.format(total_acc_1))
        # total_loss, total_acc_2 = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l2, device = device, stop = True, num_stop=params.num_stop, restarts = res, epsilon = 0.5, num_iter = params.num_iter, alpha = params.alpha_l2)
        # print('Test Acc 2: {0:.4f}'.format(total_acc_2))
        # total_loss, total_acc_inf = epoch_adversarial(test_batches, None, model, epoch_i, pgd_linf, device = device, stop = True, num_stop=params.num_stop, num_iter = params.num_iter, restarts = res)
        # print('Test Acc Inf: {0:.4f}'.format(total_acc_inf))
    elif params.pgd_norm == 11:
        for i in range(13):
            step_alpha = params.alpha + i*0.1
            total_loss, total_acc_l1_sign_free = epoch_adversarial(test_batches, None, model, epoch_i, pgd_l1_sign_free, device = device, stop = params.test_subset, num_stop=params.num_stop, restarts = res, alpha = step_alpha, num_iter = params.num_iter)
            print('Alpha: {:.4f}. Test Acc L1 signfree: {:.4f}'.format(step_alpha, total_acc_l1_sign_free))
        for i in range(5):
            step_alpha = params.alpha_topk + i*0.1
            total_loss, total_acc_1 = epoch_adversarial(test_batches, None,  model, epoch_i, pgd_l1_topk, device = device, stop = True, num_stop=params.num_stop, restarts = res, num_iter = params.num_iter, alpha=step_alpha)
            print('Alpha: {:.4f}. Test Acc 1: {:.4f}'.format(step_alpha, total_acc_1))

def fast_adversarial_DDN(model_name):
    #Saves the minimum epsilon value for successfully attacking each image via PGD based attack as an npy file in the folder corresponding to model_name
    #No Restarts
    #Done for a single batch only since batch size is supposed to be set to 1000 (first 1000 images)
    print (model_name)
    print(device)
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2, gpu_id = torch.cuda.current_device())
    model = PreActResNet18().to(device)
    for m in model.children(): 
        if not isinstance(m, nn.BatchNorm2d):
            m.float()   

    model.load_state_dict(torch.load(model_name+".pt", map_location = device))
    model.eval()
    
    for i,batch in enumerate(test_batches): 
        x,y = batch['input'].float(), batch['target']
        restarts = 1        
        min_norm = np.zeros((restarts, batch_size))
        for i in range(restarts):
            try:
                attacker = DDN(steps=100, device=device)
                adv = attacker.attack(model, x, labels=y, targeted=False)
            except:
                attacker = DDN(steps=100, device=device)
                adv = attacker.attack(model, x, labels=y, targeted=False)
            delta = (adv - x)
            norm = norms(delta).squeeze(1).squeeze(1).squeeze(1).cpu().numpy() 
            min_norm[i] = norm
        min_norm = min_norm.min(axis = 0)
        np.save(model_name + "/" + "DDN" + ".npy" ,min_norm) 
        break



def test_saver(model_name):
    #Saves the minimum epsilon value for successfully attacking each image via PGD based attack as an npy file in the folder corresponding to model_name
    eps_1 = [3,6,(2000/255),12,20,30,50,60,70,80,90,100]
    eps_2 = [0.05,0.1,0.2,0.3,0.5,0.7,1,2,3,4,5,10]
    eps_3 = [0.005,0.01,(4/255),0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,1]
    num_1 = [50,50,100,100,100,200,200,200,300,300,300,300]
    num_2 = [30,40,50,50,100,100,150,150,150,150,300,300]
    num_3 = [30,40,50,50,100,100,150,150,150,150,300,300]

    attacks_l1 = torch.ones((batch_size, 12))*1000
    attacks_l2 = torch.ones((batch_size, 12))*1000
    attacks_linf = torch.ones((batch_size, 12))*1000
    model = PreActResNet18().cuda()
    # for m in model.children(): 
    #     if not isinstance(m, nn.BatchNorm2d):
    #         m.half()   
            
    model_address = model_name + ".pt"
    model.load_state_dict(torch.load(model_address, map_location = device))
    criterion = nn.CrossEntropyLoss()

    model.eval()        
    test_batches = Batches(test_set, batch_size, shuffle=False, gpu_id = device_id)

    try:
        total_loss, total_acc = epoch(test_batches, None, model, 0, criterion, opt = None, device = device, stop = True)
    except:
        print ("OK")

    for index in range(len(eps_1)):
            e_1 = eps_1[index]
            n_1 = num_1[index]
            eps, total_acc_1 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l1_topk, e_1, n_1, device = device, restarts = res)
            attacks_l1[:,index] = eps
    attacks_l1 = torch.min(attacks_l1,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL1" + ".npy" ,attacks_l1.numpy())

    for index in range(len(eps_2)):        
            e_2 = eps_2[index]
            n_2 = num_2[index]
            eps, total_acc_2 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l2, e_2, n_2, device = device, restarts = res)
            attacks_l2[:,index] = eps
    attacks_l2 = torch.min(attacks_l2,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL2" + ".npy" ,attacks_l2.numpy())

    for index in range(len(eps_3)):
            e_3 = eps_3[index]
            n_3 = num_3[index]
            eps, total_acc_3 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_linf, e_3, n_3, device = device, restarts = res)
            attacks_linf[:,index] = eps
    attacks_linf = torch.min(attacks_linf,dim = 1)[0]
    np.save(model_name + "/" + "CPGDLINF" + ".npy" ,attacks_linf.numpy())
  


model_list = ["LINF", "L1", "L2", "MSD_V0", "TRIPLE", "WORST", "VANILLA"]
model_name = "Selected/{}".format(model_list[choice])
if path is not None:
    model_name = path

import os
if(not os.path.exists(model_name)):
    os.makedirs(model_name)
    

print (model_name)
if attack == 0:
    test_foolbox(model_name, 1000)
elif attack == 1:
    test_pgd(model_name)
elif attack ==2:
    test_saver(model_name)
elif attack == 3:
    fast_adversarial_DDN(model_name)