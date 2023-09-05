# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class de_cnn(nn.Module):
    def __init__(self):
        super(de_cnn, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, 4, stride=2, bias=False, padding=0),
            nn.BatchNorm1d(32),
            nn.ELU()
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 16, 4,  stride=2, bias=False, padding=0),
            nn.BatchNorm1d(16),
            nn.ELU()
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 8, 4,  stride=2, bias=False, padding=1),
            nn.BatchNorm1d(8),
            nn.ELU()
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 4, 4,  stride=2, bias=False, padding=1),
            nn.BatchNorm1d(4),
            nn.ELU()
            )                           
                              
                
        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(4, 8, 4,  stride=2, bias=False, padding=1),
            nn.BatchNorm1d(8),
            nn.ELU()
            )
        
        
        self.layer6 = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 4,  stride=2, bias=False, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU()
            )

        self.layer7 = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 4,  stride=2, bias=False, padding=0),
            nn.BatchNorm1d(32),
            nn.ELU()
            )

        self.layer8 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, 4,  stride=2, bias=False, padding=0),
            nn.BatchNorm1d(1),
            nn.ELU()
            )            
                                                    
                           
    def forward(self, x):
                                                                                               
        
        L1 = self.layer1(x)

        L2 = self.layer2(L1)
        
        L3 = self.layer3(L2)
        
        L4 = self.layer4(L3)
                
        L5 = self.layer5(L4)
                                                                            
        L6 = self.layer6(L5+L3)
                        
        L7 = self.layer7(L6+L2)
                        
        L8 = self.layer8(L7+L1)
                                                                                            
        return  L8, L3
    
class de_finetune(nn.Module):
    def __init__(self, net = de_cnn):
        
        super(de_finetune, self).__init__()
    
        self.encoder = net
                
        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(32, 16, 4,  stride=2, bias=False, padding=0),
        #     nn.BatchNorm1d(16),
        #     nn.ELU(inplace=False)
        #     )        
                          
        # self.layer3 = nn.Sequential(
        #     nn.Conv1d(16, 8, 4,  stride=2, bias=False, padding=1),
        #     nn.BatchNorm1d(8),
        #     nn.ELU(inplace=False),
        #     )
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 4, 4,  stride=2, bias=False, padding=1),
            nn.BatchNorm1d(4),
            nn.ELU(inplace=False)
            )
        
        self.classifier = nn.Linear(4*42, 20)        
                    
    def forward(self, x):
        
        encoder = self.encoder
        
        x1, x2 = encoder(x)
        
        # layer2 =  self.layer2.cuda()
                
        # layer3 =  self.layer3.cuda()
        
        layer4 =  self.layer4.cuda()
        
        # x2 = layer2(x2) 
        
        # x2 = layer3(x2)
        
        x2 = layer4(x2)
        
        x2 = x2.view(x2.size(0), -1)   
                                
        classfier = self.classifier.cuda()
                
        out = classfier(x2)
                
        return out

# if __name__=='__main__':

#     model1 = de_cnn().cuda()
    
#     model2 = de_finetune(model1)
                    
#     x = torch.randn(1,1,678).cuda()
                    
#     a, b = model1(x)    
    
#     c = model2(x)    
      
# print('reconstruct_size',a.shape, b.shape, c.shape)