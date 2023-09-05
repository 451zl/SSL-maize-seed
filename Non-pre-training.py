import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from dataset import BatchData
import scipy.io as io
from net import de_cnn
from net import de_finetune
from tqdm import tqdm
import torch.optim as optim
from sklearn import metrics


##load data                    
##
train = io.loadmat('./data/train.mat')
train = train['train']

label_trian = io.loadmat('./data_processing/label_train.mat')
label_trian = label_trian['label_train']

test = io.loadmat('./data/test.mat')
test = test['test']

label_test = io.loadmat('./data_processing/label_test.mat')
label_test = label_test['label_test']

###
train = train.reshape(train.shape[0], 1, train.shape[1])

test = test.reshape(test.shape[0], 1, test.shape[1])


train_loader = DataLoader(BatchData(train, label_trian),
                         batch_size = 128,
                         shuffle=True)

test_loader = DataLoader(BatchData(test, label_test),
                         batch_size = 2000,
                         shuffle=False)

##load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = de_cnn().to(device)

model2 = de_finetune(model)
optimizer=torch.optim.Adam(model2.parameters(), lr=0.001,betas=(0.9, 0.999), weight_decay=1e-6)


epoch_n = 1000
criterion = nn.CrossEntropyLoss()


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
            for i, (x,y) in enumerate(tqdm(train_loader)):
                

                x = x.to(device)
                x = x.type(torch.cuda.FloatTensor)
                                                                                                                                                
                y = y.to(device)                
                y = y.type(torch.cuda.LongTensor)   
                                                                               
                outputs = model2(x)                                                                                                                                                                                                             
                loss_train = criterion(outputs, y.squeeze())                                                                                                                                      
                # BP
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                losses_test.append(loss_train.item()) 
                                
                pred =outputs.argmax(dim=-1)
                correct += sum(pred == y.squeeze()).item()
                wrong += sum(pred != y.squeeze()).item()
                acc = correct / (wrong + correct)        

            print("Train Acc: {}".format(acc*100))
            print("train loss :", '{:.16f}'.format(torch.mean(loss_train))) 
            with torch.no_grad():
                    
                for i, (x1, y1) in enumerate(test_loader):                
                                                                                    
    
                    x1 = x1.to(device)
                    x1 = x1.type(torch.cuda.FloatTensor)
                                                                                                                                                    
                    y1 = y1.to(device)                
                    y1 = y1.type(torch.cuda.LongTensor)   
                                                                                                           
                    outputs_t = model2(x1)                                                                                                                                                 
                    
                    valid_loss =  criterion(outputs_t, y1.squeeze())                    
                                        
                    pred_t = outputs_t.argmax(dim=-1)
                    correct_t = sum(pred_t == y1.squeeze()).item()
                    wrong_t = sum(pred_t != y1.squeeze()).item()
                    acc_test = correct_t / (wrong_t + correct_t)
                    
                    kappa = metrics.cohen_kappa_score(pred_t.cpu(), y1.cpu(), labels=None, weights=None, sample_weight=None)
                                    
                                  
            print("test loss :", np.mean(losses_test))
            print("Test Acc: {}".format(acc_test*100))  
            print('kappa:', kappa)                