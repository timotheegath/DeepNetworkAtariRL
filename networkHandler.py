#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:00:08 2017

@author: timothee
"""

import pickle
import torch.nn as nn


def saveNetwork(DQN):
    
    dictionary = DQN.state_dict()
    pickle.dump(dictionary, open("network.p","wb"))

def loadNetwork(DQN):
    
    data=pickle.load(open("network.p","rb"))
    DQN.load_state_dict(data)
    