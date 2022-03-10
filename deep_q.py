# Virtual Self-driving Car using deep Q-learning
# ENPM690 Rahul Karanam
# Importing necessary libraries

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Now we create the base(network) for our deep learning model(dlm)


# Class : Dlm
# input_size [input] - The input size for the given model. (Number of inputs)
# action_n  [input] - Number of actions. ( Number of Outputs)
# forward method : This method will forward propagate the signal inside
#                  the network and returns the predicted q-values.
# q-values [output]
# We are calling the nn.Module class to access all the tools and properties of the module class.

class Dlm(nn.Module):

    def __init__(self, input_size , action_n):
        super(Dlm,self).__init__() # we use super to activate the inheritance of module class 
        self.input_size = input_size
        self.action_n=action_n
        self.fc1 = nn.Linear(input_size,40) # Creating a fully connected lasample
        self.fc2 = nn.Linear(40,action_n)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# We create a class for storing our transitions for experience replay buffer
# capacity [input] is the maximum size of the memory. 
# event [input] - this is the state transition (s,a,r,s_next) to our memory
# batch_size : It is size of batch of the state transitions.
# This class will output a sample transitions from the replay buffer for training.

class ExperienceReplay(object):

    def __init__(self,capacity):
        self.capacity=capacity
        self.memory = []

    def push(self,event):       # If the memory reaches its capacity we need to delete the first element in the memory
        self.memory.append(event)
        if len(self.memory) > self.capacity: 
            del self.memory[0]
    # method: sample_transition
    # This method gives you a mini-batch of state tranisitons as (prev_state,action,reward,next_state) based upon
    # the number of batch size.
    # we use pytorch variable class as it allows fast computation during mini-batch gradient descent


    def sample_tranisiton(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)



# Implementing the Deep Q-Learning Model

# action_state 
# action_n 
# gamma - Discount Factor to be used in computing the Temporal Difference
# Adadelta is our optimizer which updates the weights through mini-batch gradient descent
# during backpropagation.
# self.last_state is used to match each last state to the appropriate batch.


class DeepQ(object):


    def __init__(self,input_size,action_n,gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Dlm(input_size,action_n)
        self.memory = ExperienceReplay(capacity = 100000) # Initial memory size 
        self.optimizer = optim.Adadelta(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # We unsqueeze to give it as a input to the softmax for prediction of action.
        self.last_action = 0  #last action played at each epoch
        self.last_reward = 0 # last reward after the last action from the state
      
    # This method will select action for each iteration using the softmax
    # taking the q-values as input from the model which is a object of 
    # our Dql class which calls the forward method.
    # state [input] : [ orientation , signal_1, signal_2, signal_3]

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state))*100)
        action = probs.multinomial(len(probs))
        return action.data[0,0]
    
    # This method will be used to implement the q-learning, update the weights 
    # where the value of back-propagation is least or the loss is decreasing at 
    # each iteration.

    # batch_states  [input]: A batch of input states.
    # batch_actions [input]: A batch of actions played.
    # batch_rewards [input]: A batch of the rewards received.
    # batch_next_states [input]: A batch of the next states reached.

    # batch__outputs = Q(state_time_batch,action_time_batch) the batch of outputs at a iteration t
    # batch_targets = rewards(state_time_batch,action_time_batch)+ gamma*max(Q(next_state_time_batch,action)) the batch of targets
    # Temporal difference(td) is the difference between batch_outputs and batch_targets
    # Loss is 0.5*(td)^2, i.e sum of squared differences between the batch_outputs and batch_targets.


    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)
        self.optimizer.zero_grad() # this will set all the gradients to zero
        td_loss.backward() # backpropagate the error loss (td_loss) 
        self.optimizer.step() # updates the weights using the adadelta optimizer
    
    # This method will take the new_state and new_reward as the input
    # after playing an action, returns the new action to perform after performing
    # the weight updates.
    
    # We take our new_state, which we will push it to our memory, we select a new action
    # based upon the new_state and check whether the memory has reached 100 batches, 
    # if reached we call the learn method to learn the new state parameter and update
    # it using the loss function.

    # We return the new_action along with the last state,action and reward for the previous
    # state.

    
    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float().unsqueeze(0)
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state))
        new_action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample_tranisiton(100)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return new_action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
