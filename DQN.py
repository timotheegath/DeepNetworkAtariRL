#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:51:32 2017

@author: timothee
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools



use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class DQN(nn.Module):#Working, all layer sizes have been checked
    def __init__(self,variableSize, possibleButtonSize): #It will be necessary to know how many variables there are to adjust the length 
    #of the linear layer exclusive to variable data
        
    
    ##############Network definition##############
        super(DQN,self).__init__()
        self.vSize = variableSize
        
        self.actionMap = list(itertools.product([0,1],repeat = possibleButtonSize)) #from https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
        
        #Cleanup actions involving pressing more than two buttons at once
            
        self.actionMap = [x for x in self.actionMap if sum(x) < 3]
              
        
        #First, the shared convolutional layers, only for graphical data
        self.conv1 = nn.Conv2d(8, 32, kernel_size=7, stride=2, padding = 2)    #Replace with Atari specs    
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding = 2)
        self.max1 = nn.MaxPool2d(3, stride = 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding = 2)
        self.max2 = nn.MaxPool2d(3, stride = 2)
        self.conv4 = nn.Conv2d(128, 192, kernel_size=3, stride=2, padding = 2)
        #Exclusive linear layer for variable data, make it the same size as the data (recommendation of Kai)
        self.linV = nn.Linear(variableSize,variableSize)        
        #Specific head: attacking        
        self.interA = nn.Linear(1728 + variableSize,128)#512
        self.headA = nn.Linear(128, len(self.actionMap)) #Output size: explanation below in action map
#        self.headA = nn.Linear(64, pow(2,1))
        #Specific head: pack gathering 
        self.interP = nn.Linear(1728 + variableSize,128)
        self.headP = nn.Linear(128, len(self.actionMap))
        

        #Classifier head
        self.headC = nn.Linear(1728 + variableSize,1)       
        
        
        ######Action map#########
        #This is necessary since VizDoom allows multiple actions to be taken at the same time
        #To perform an action, VizDoom takes in an array of the size of all available buttons
        #any value other than 0 means activate
        #Now, the action to be taken doesn't match the index of the max Q value
        #We create a map that contains any possible combination of all the available buttons
        #We have as many output neurones as there are binary array combinations
        
        
        
        
        
    def forward(self,state):
        #split the graphical data from the variable data (should not go through conv)
        x = state[:,0:8,:,:]   #NOTE: 0:8 excludes 8 !!! only index from 0 to 7     
        y = state[:,8,:,:].contiguous().view(state.size(0),-1)#extract the variable tensor and make it linear
        y = y[:,0:self.vSize] #only keep the relevant variables
        
        #pass through conv
        x = F.relu(self.conv1(x))
        x = F.relu(self.max1(self.conv2(x)))
        x = F.relu(self.max2(self.conv3(x)))
        x = F.relu(self.conv4(x))
    
        x = x.view(x.size(0),-1)
        
        
        
        #in the mean time, the variables pass through their own layer
        y = F.relu(self.linV(y))
        
        #join the output of the conv with the variable data again
        xy = torch.cat([x,y],1)
      
        #Attack head
        a = F.relu(self.interA(xy))
        a = self.headA(a)
        
        #Pack gathering head
        p = F.relu(self.interP(xy))
        p = self.headP(p)
        
        #Classifier head
        c = F.relu(self.headC(xy))
        
        
        
        #Put both heads in tuple
      
        
        return a,p,c

    def indexInterpreter(self,index):#maps the index of the max Q value to an array of button activation
                
        
        return list(self.actionMap[index])