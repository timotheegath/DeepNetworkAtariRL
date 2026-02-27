#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:57 2017

@author: timothee
"""
import random
from collections import namedtuple

Transition = namedtuple('Transition',('state','action','next_state','reward'))
#identical to Catch
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory = []
        self.position=0
        
    def push(self,*args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1)% self.capacity
    
    def sample(self, batch_size):
        samples=random.sample(self.memory , batch_size)
        
        return samples
    
    def __len__(self):
        return len(self.memory)
    
    
