# VIRTUAL SELF-DRIVING CAR USING DEEP-Q-LEARNING

### Introduction
This project explains the use case of a Reinforcement learning technique called Q-learning.
Deep Q-learning
In more formal terms, an agent in an environment with a specified state can perform a set
of actions. It receives a reward after executing such activities, which indicates how effective
that action was. Of course, we want to be rewarded with the best possible rewards for
achieving our goal. 

The Bellman Equation is used to account for future benefits because
they are usually the result of a series of activities. We use these rewards to update Q-Values
in Q-Learning, which inform us how good/desirable a particular condition is. Rather than
storing Q-Values, we use a Deep Neural Network in Deep Q-Learning to approximate
Q-Values using our current state as input.


## Packages Needed

```python
numpy
Pytorch
Kivy
Matplotloib
```


## Steps to Run

```python

$ git clone --recursive https://github.com/karanamrahul/Reinforcment-Learnings-Projects.git

$ cd Virtual_car_q_learning

$ python3 environment.py

$ python3 deep_q.py

# This will pop up a kivy window.
# Start drawing roads and wait for the robot to learn and drive autonomously.
```



![](https://github.com/karanamrahul/Reinforcment-Learnings-Projects/blob/main/results/self_drive.gif)

![](https://github.com/karanamrahul/Reinforcment-Learnings-Projects/blob/main/results/selfdrive.gif)


### Deep Q-learning model process

1. First we initialize the memory of the experience replay buffer to an empty list with
the maximum capacity ( N = 100000 for our case).

2. We implement a deep neural network model with 2 layers , the input layers consist
of input_size number of input layers and 40 hidden layers ( neurons ) which are
connected to a fully connected layer of (40 , action_n) which is the number of
actions.

3. This neural network will predict the q-values which is later used to predict the action
using softmax function which calculates the max probability of those q-values.

4. After playing the action based upon the prediction , we compute the reward R(s,a)
given the current state (s) and action (a) at each iteration. For every action, we add
these state transitions to our experience replay buffer.

5. State_transisiton : (prev_state,action,reward,next_state)
 
6. After performing these actions and appending all the state transitions to our buffer,
we then sample a random batch of state transitions from our buffer and use it for
training our neural network model.

7. We give the batch to the model and generate predictions i.e q-values
Q(state_batch,action_batch) along with we compute the batch targets.

8. We then find the loss function computed using the temporal difference.Temporal
difference (TD) is the difference between the targets and predicted values(batches
as we are giving batches of data to our neural network). Loss function will sum of
squared difference between these values.

9. After computing the loss function , we then backpropagate the loss using the
pytorch. We have used AdaDelta optimizer to update the weights using mini-batch
gradient descent for every iteration of the input batch.

```
Input States
[orientation,sensor-1,sensor-2,sensor-3]
Output Actions
rotations=[forward,left , right]
```

```
Environment
I have used kivy as my environment in order to show my agent.
```

#### Neural Network Model
Input Layer: 4 Nodes (One for each state-input)

Hidden Layer: 40 Nodes

Output Layer: 3 Nodes (One for each action)

Activation Functions: ReLU

Optimizer: AdaDelta

### References

https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-intoreinforcement-learning.html

@credits : https://github.com/PacktPublishing/AI-Crash-Course
https://www.thinkautonomous.ai/blog/?p=deep-reinforcement-learning-for-self-driving-cars-an-intro3


