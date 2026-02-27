#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:11:01 2017

@author: timothee
"""
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch


#usual cuda check
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


#T.Compose creates a succession of torch vision transforms. All of them can take either an image or numpy array in. T.ToTensor will ouput a tensor  
ctrCrop = T.Compose([T.CenterCrop(128),T.Scale(120),T.ToTensor()]) #ctrCrop wil involve taking a square of 128 in the middle followed 
#by a scaling to 64X64
overallCrop = T.Compose([T.CenterCrop(240),T.Scale(120),T.ToTensor()]) #similar,but takes a square overall crop.
# 240 is the height of the screen, smaller dimension than the width




def grayscale(original):#Tested, works
    #Check if input is already numpy
    if type(original) is np.ndarray :
        
        new = adaptShape(original)#necessary for the dot operation to work, the RGB dim must be in first position
        new = np.dot(new[...,:3], [0.299, 0.587, 0.114])
        new = np.ascontiguousarray(new, dtype = np.int8)
        return new#Using previous technique from Catch
    
    else:#make it numpy and convert back to image
        
        new = np.asarray(original)
        new = adaptShape(new)
        new = np.dot(new[...,:3], [0.299, 0.587, 0.114])
        new = np.ascontiguousarray(new, dtype = np.int8)
        
        return Image.fromarray(new)


def doCenter(original):#Tested,works
    if type(original) is np.ndarray :
        
        new = Image.fromarray(adaptShape(original))
    
        
    return ctrCrop(grayscale(new)).type(FloatTensor)/255

def doOverall(original):#Tested, works

    if type(original) is np.ndarray :

        new = Image.fromarray(adaptShape(original))
        
    return overallCrop(grayscale(new)).type(FloatTensor) /255 
    
    


    
def show(im):#Only for troubleshooting
    
    if type(im) is np.ndarray :
        
        Image.fromarray(im).show()
        
    else:
        im.show()
        
def adaptShape(original):#necessary to put the RGB dimension in first position like in Catch
    
    return np.moveaxis(original, 0, -1)#shift all the axis -by -1 > the axis 0 will find itself in position -1

