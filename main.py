#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:00:14 2017

@author: timothee
"""

from vizdoom import *
import random
import time

import graphics
import torch
from collections import deque
from collections import namedtuple
from DQN import DQN
import networkHandler as nH
from torch.autograd import Variable
from memory import ReplayMemory
from copy import deepcopy

import torch.optim as optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from IPython import display

################## Initialize game ############################################

ticks = 0
game = DoomGame()
game.load_config("scenarios/custom.cfg")

game.init()
game.send_game_command('sv_infiniteammo 1')
game.set_death_penalty(500)

################## Initialize tuples and tensors ##############################

Transition = namedtuple('Transition',('state','action','next_state','reward'))
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
Tensor = FloatTensor

################## Initialize learning variables ##############################

EPISODES = 700
MEM_CAPACITY = 100000
EPSILON_START = 0.95
EPSILON_END = 0.05
EPSILON_DECAY = 1000
BATCH_SIZE = 64
GAMMA = 0.9
LR = 0.00005
epsilon = EPSILON_START
current_scenario = 0
previous_scenario = 0
liv_penalty = -0.4
variableSize = game.get_available_game_variables_size() #get how many variables are available for later use
possibleButtonSize = game.get_available_buttons_size() #get how many buttons are available

################## Initalize learning tools ###################################

mem1 = ReplayMemory(MEM_CAPACITY)
mem2 = ReplayMemory(MEM_CAPACITY)
mems = (mem1,mem2)
model = DQN(variableSize,possibleButtonSize)#pass the number of variables to the DQN
target = DQN(variableSize,possibleButtonSize)

if use_cuda:
    model.cuda()
    target.cuda()

optimizer=optim.RMSprop(model.parameters(),lr=LR)

################## Class definitions ##########################################

class Buffer():#Creates a buffer class with inherent ability of appending new data and resetting itself
    #identical with Catch
    def __init__(self):
        self.buffer = deque([], maxlen=4)
        for i in range(4):
            self.buffer.append(Tensor(1, 120, 120).zero_())
    
    def resetBuffer(self):
        
        for i in range(4):
            self.buffer.append(Tensor(1, 120, 120).zero_())
            
    def add(self,newState):
        self.buffer.append(newState)
       
################## Class declarations ######################################### 
  
centerBuffer = Buffer() #Initialize both buffers globally
overallBuffer = Buffer()

################## Input manipulation functions ###############################

def graphicInput(VizState):#Extracts the graphical information from the state info that VIzdoom returns
    
    screen = VizState.screen_buffer
    centerBuffer.add(graphics.doCenter(screen)) #append the center view into the buffer
    overallBuffer.add(graphics.doOverall(screen)) #append the overall view into the buffer
    #NOTE: no need to unsqueeze the tensors since the former RGB dimension is still present
    
def variableInput(VizState):#Extracts the variable information
    
    varTens = Tensor(120,120).zero_() #Initalize a tensor filled with zeros of the right size
    variables = Tensor(VizState.game_variables)
    
    for i in range(variableSize): #Replace as many zeros as needed by a variable value until there are no more variables to put in
        varTens[i] = variables[i]  
 
    return varTens.unsqueeze(0) #Add the usual 3D dimension

def assembleState(VizState):#Tested, working     Converts the state info returned by Doom into a state viable for our DQN: 9 X 64 X 64 Tensor
    
    graphicInput(VizState) #Update the buffers
    variables = variableInput(VizState) #Update the variable tensor       
    state_temp = list(centerBuffer.buffer) + list(overallBuffer.buffer) + [variables] #Join them in a sequence of tensors    
    state = torch.cat(state_temp,0) #Concatenate them into a 9 X 64 X 64 tensor
    state.unsqueeze_(0) #add the batch dimension
   
    return state

################## Game flow functions ########################################
    
def selectAction(state):#Tested,working
    global epsilon
    
    if epsilon >= EPSILON_END:#epsilon value update
        epsilon = EPSILON_START - EPSILON_END/EPSILON_DECAY*ticks 
        
    sample = random.random()
    
    if sample > epsilon:
        
        action_index = model(Variable(state, volatile=True).type(FloatTensor))[current_scenario].data.max(1)[1].view(1, 1) #get the index of the output with the highest predicted Q-value
        action = model.indexInterpreter(action_index[0,0])#Translate this index into a combination of button activation
        
        return action,action_index #Need to return both, action for VizDoom to understand, and action_index for the memory and later training
    
    else:
        
        action_index = LongTensor([[random.randint(0, len(model.actionMap)-1)]])
        action = model.indexInterpreter(action_index[0,0]) #Translate this index into a combination of button activation
        
        return action,action_index

def selectAction_noOracle(state):
    out = model(Variable(state, volatile=True).type(FloatTensor))
    
    current_scenario = round(out[2].data[0,0])
    if current_scenario>1:#Preventing the classifier from being any other value than 0 or 1
            current_scenario = 1
    action_index = out[current_scenario].data.max(1)[1].view(1, 1) #get the index of the output with the highest predicted Q-value
    action = model.indexInterpreter(action_index[0,0])#Translate this index into a combination of button activation
        
    return action, action_index, out #Need to return both, action for VizDoom to understand, and action_index for the memory and later training
    
    
def optimize_model():
    #Check if memories have enough material to fill up a batch    
    if (len(mems[0]) < BATCH_SIZE)|(len(mems[1]) < BATCH_SIZE):
        return #get out 
    
    i = 0 #Iterator to select right branch (0 or 1)
    lossC = 0 #Will store the loss function of the classifier
    loss = 0 #Will store the loss function of the heads
    
    for memory in mems:
        transitions = memory.sample(BATCH_SIZE)       
        batch = Transition(*zip(*transitions))        
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))      
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]).cuda(),
                                         volatile=True)
        #Assemble batches
        
      
        state_batch = Variable(torch.cat(batch.state).cuda())
        action_batch = Variable(torch.cat(batch.action).cuda())
        reward_batch = Variable(torch.cat(batch.reward).cuda())

        
        #LOSS OF CLASSIFIER
        classification = model(state_batch)[2] #Predicted classification according to the model
        
        targetC = Variable(Tensor([[i]]).cuda()) #i is what the classification should actually be, make it a variable
        
        targetC = targetC.expand_as(classification) #make it match the shape of the classifier output of the model
        lossC += F.smooth_l1_loss(classification,targetC) #Build the , add it so that
        #the loss of the classification of both minibatches will be processed
        
        #LOSS OF HEADS
        state_action_values = model(state_batch)[i].gather(1, action_batch) #Prediction, what the expected Q-value is by following policy 
        
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))        
        next_state_values[non_final_mask] = target(non_final_next_states)[i].max(1)[0] #Target, what the max Q-value at the next state actually is 
        next_state_values.volatile = False
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch #so if following the policy, 
        #the value of the current state will be the reward you get by following the policy + the value of the next state (discounted)
        
        loss += F.smooth_l1_loss(state_action_values, expected_state_action_values)#Build the loss, add it to the one of the previous branch
        i=i+1 #next branch
    

            
    optimizer.zero_grad()

    (loss + lossC).backward()
    
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
       
    
def oracle():
    global current_scenario
    global previous_scenario
    health = game.get_game_variable(GameVariable.HEALTH)
    ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
    previous_scenario = current_scenario
    if (health < 50):
        
        current_scenario = 1
        
    
    else :
        
        current_scenario = 0
        
        
    
       
def changeConditions(silent = False):##Make it more rewarding to stay alive when under 50% health
    global current_scenario
    global previous_scenario
    if previous_scenario!=current_scenario:
        if not silent :
            print('scenario changed')
        
        if current_scenario == 0:
            game.set_living_reward(liv_penalty)
            if not silent :
                print('Changed back to healthy scenario, last reward: ', game.get_last_reward(), " and health :", game.get_game_variable(GameVariable.HEALTH)) #for debug purposes:
        else :
            game.set_living_reward(-1 * liv_penalty) 
            
        

        
#############################     TESTING FUNCTIONS ###########################
def PerformanceTest():
    global current_scenario
    global previous_scenario
    storeDefTimeout = game.get_episode_timeout()
    game.set_episode_timeout(10000)
    Q_values = []
    game.new_episode()
    state = assembleState(game.get_state())
    current_scenario = 0
    previous_scenario = 0
    while not game.is_episode_finished():
        
        previous_scenario = current_scenario
        action,_,out = selectAction_noOracle(state)
        current_scenario = out[2].data[0,0]
        
        changeConditions(silent = True) #Still necessary to make living more profitable when under 50% health
        Q=(out[0].max(1)[0].data[0],
                   out[1].max(1)[0].data[0],
                   out[2].data[0,0])
        game.make_action(action,3)
        
        if not game.is_episode_finished():
            state = assembleState(game.get_state())
        Q_values.append(Q)
        
    
    game.set_episode_timeout(storeDefTimeout)    
    return game.get_total_reward(), Q_values   

score = []       
def plotValues():
    
    
    global score
   
    
    net_score, Q_values= PerformanceTest()
    
    
    score.append(net_score)    
    plt.ioff()#off display
    plt.figure()
    plt.clf()
    
    
    
    plt.subplot(311)
    plt.plot(score)
    plt.xlabel('Episode number')
    plt.ylabel('Net score')
    
    
    if len(Q_values) > 1:
        plt.subplot(312)
            
        Q_values=tuple(zip(*Q_values))
        plt.plot(Q_values[0],label = 'Attack branch')
        plt.plot(Q_values[1], label = 'Low health branch')
        
        plt.xlabel('Frame number in test sequence')
        plt.ylabel('Q Values') 
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        plt.subplot(313)
        plt.plot(Q_values[2], label = 'Classifier')
        plt.xlabel('Frame number in test sequence')
        plt.ylabel('Classification')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    display.clear_output(wait=True)
    fig=plt.gcf()
    
    fig.set_size_inches(8, 15)
    fig.savefig('graphs/graph.PNG')

def testNetwork():
    global model
    centerBuffer.resetBuffer()
    overallBuffer.resetBuffer()
    plotValues()
    
#############################     GAME LOOP   ##############################################################"
for i in range(EPISODES):
    
    game.new_episode()
    state = assembleState(game.get_state())
    current_scenario = 0
    previous_scenario = 0
    while not game.is_episode_finished():
        

        oracle()
        changeConditions()
        action = selectAction(state) 
        
        reward = game.make_action(action[0],3) #pass action[0], understanble by Vizdoom
       
        reward = Tensor([reward]) #wrap reward in Tensor for optimizer
        if game.is_episode_finished():
            next_state = None
        else :
            next_state_gpu = assembleState(game.get_state()) #Can't set a none type to cpu
            next_state = next_state_gpu.cpu()
        
        mems[current_scenario].push(state.cpu(),action[1].cpu(),next_state,reward.cpu())# Store action[1], understandable by Pytorch
        state = next_state_gpu
		# print('action:',action[0])
        if(ticks%3 == 0):
            optimize_model() ##Only optimize every three ticks
            
        
        
        time.sleep(0.02)        
        ticks += 1
        
        if(ticks % 100==0):
            target.load_state_dict(deepcopy(model.state_dict()))#update target
    print ("Result:", game.get_total_reward())
    time.sleep(2)
    testNetwork()
    centerBuffer.resetBuffer()
    overallBuffer.resetBuffer()
    nH.saveNetwork(model)
    
