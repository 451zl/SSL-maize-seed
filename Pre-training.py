# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import BatchData
from torch.utils.data import Dataset,DataLoader
import scipy.io as io
from net import de_cnn
from tqdm import tqdm
import torch.optim as optim
import time
##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = de_cnn().to(device)
##导入数据            
train = io.loadmat('./data/train.mat')
train = train['train']

train_s = io.loadmat('./data/train_ns.mat') ##nosie data (SNR=40db)
train_s = train_s['train_ns']

test = io.loadmat('./data/test.mat')
test = test['test']

test_s = io.loadmat('./data/test_ns.mat')
test_s = test_s['test_ns']
# ##%%
train = train.reshape(train.shape[0], 1, train.shape[1])
train_s = train_s.reshape(train_s.shape[0], 1, train_s.shape[1])

test = test.reshape(test.shape[0], 1, test.shape[1])
test_s = test_s.reshape(test_s.shape[0], 1, test_s.shape[1])

train_loader = DataLoader(BatchData(train, train_s),
                         batch_size = 128,
                         shuffle=True)

test_loader = DataLoader(BatchData(test, test_s),
                         batch_size = 2000,
                         shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), weight_decay=1e-6)

print(optimizer)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

epoch_n = 10
criterion = nn.MSELoss()
start = time.perf_counter() 

correct = 0
wrong = 0
correct_t = 0
wrong_t = 0  
 
for epoch in range(epoch_n):
        
            loss_train = []
            losses_test = []
            print("Epoch {}/{}".format(epoch+1,epoch_n))
            print("-"*10)    
            # training                   
            for i, (spe, spe_ns) in enumerate(tqdm(train_loader)):
                
                inputs_spe = spe[:,:,0:678]  
                inputs_spe = inputs_spe.to(device)
                inputs_spe = inputs_spe.type(torch.cuda.FloatTensor)
                                                                                                                 
                inputs_spe_ns = spe_ns[:,:,0:678]               
                inputs_spe_ns = inputs_spe_ns.to(device)
                inputs_spe_ns = inputs_spe_ns.type(torch.cuda.FloatTensor)  
                                               
                outputs,outputs_1 = model(inputs_spe_ns)                                                                                                                                                                                                   
                loss_train  = criterion(outputs, inputs_spe)
                                                                                                                                          
                #BP
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                losses_test.append(loss_train.item()) 
                                
            print("train loss :", '{:.16f}'.format(torch.mean(loss_train))) 
            
            
            with torch.no_grad():
                    
                for i,(spe_t, spe_tns) in enumerate(test_loader):
                
                    in_spe_t =spe_t[:,:,0:678]                    
                    in_spe_t = in_spe_t.to(device)                    
                    in_spe_t = in_spe_t.type(torch.cuda.FloatTensor)  
                                                                                    
                    spe_tns =spe_tns[:,:,0:678]                    
                    spe_tns = spe_tns.to(device)                    
                    spe_tns = spe_tns.type(torch.cuda.FloatTensor)  
                    
                    outputs_t, outputs_t1= model(spe_tns)                                                                                                                                                 
                    loss_test = criterion(outputs_t, in_spe_t)                    
                    losses_test.append(loss_test.item()) 
                                        

                                
            print("test loss :", np.mean(losses_test))
   
torch.save(model.state_dict(), './model/model40.pth')                                              