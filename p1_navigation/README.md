[//]: # (Image References)

[image1]: https://github.com/safdark/DRL_course_projects/blob/master/p1_navigation/docs/screenshot_banana_bot.png "Trained Agent"
[image2]: https://github.com/safdark/DRL_course_projects/blob/master/p1_navigation/docs/screenshot_bananan_bot_graph.png "Performance Graph"
[image3]: https://github.com/safdark/DRL_course_projects/blob/master/p1_navigation/docs/screenshot_q_learning_algorithm.png "Q-Learning"

# Project 1: Navigation
![Trained Agent][image1]

## Overview

This project aims to train a Deep Reinforcement Learning Agent to navigate (and collect bananas!) in a large, square world. 

A environment give the agent a reward of +1 for collecting each yellow banana, and a reward of -1 for collecting each blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions, as discussed below.
- **`#1-2`** are the agent's velocities -- Lef/Right velocity (usually near 0) and Forward/Backward velocity (-11.2 -to- 11.2)
- **`#3-37`** contain information about the ray-based perception of objects around the agent's forward direction. There are 7 rays projecting from the agent onto the scene, at angles: [20, 90, 160, 45, 135, 70, 110], with # 90 being directly in front of the agent. Each ray has 5 fields associated with it - [Banana, Wall, BadBanana, Agent, Distance]. The first 4 fields are for each type of object that the ray can encounter on its path, whose value is set to 1 if the object is found. Finally there is a distance measure to the first object on the ray's path.

Four discrete actions are available to the agent at each step:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Given this state information, the environment's rewards, and the available actions, the agent has to learn how to best select an action at each step of the game.

The task is episodic, and in order for the environment to be considered "solved" the agent must get an average score of +13 over the **last 100 consecutive episodes**.

## Installation

1. The game runs in a Unity environment, which can be downloaded from one of the links below, according to the operating system where this is to be run:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, under the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. A python 3.5+ environment with PyTorch version 0.4.0+ must be available for the agent to run. It is recommended to use Anaconda to setup the above environment. And then, from the command prompt, activate that environment using the following commands:
```
>> cd <path-to-this-repository>
>> source activate deep-learning-python_3_7
>> jupyter notebook
```

4. It is also advisable to create an IPython Kernel to use for the Notebook.
```

```
Now, make sure that from the Jupyter Notebook, you select the above kernel by going to "Kernel --> Change Kernel --> ___"".

5. Additionally, it would be advisable if the system can make use of a GPU. If using a NVidia GPU, it is recommended to use the latest NVidia GPU driver (4.10 as of this writing).

## Implementation

The following sections break down the implementation of the DQN agent into 3 components:
- **`The Driver`**: that provides the scaffolding for the environment and the agent to interact with each other.
- **`The Q-Learning Algorithm`**: that the agent uses in conjunction with a deep neural network.
- **`Deep Q-Network (DNN)`**: the function approximator(s) that is(are) utilized by the Q-learning algorithm used by the agent.

### Driver

For this project, the IPython Notebook serves as the driver. There are 2 notebooks:
- **`Navigation-Trainer.ipynb`**: This does the training and saves the optimized parameters to a file ('checkpoint.pth')
- **`Navigation-Visualizer.ipynb`**: This allows us to visualize the performance of the agent that is loaded from the checkpoint file generated by the trainer.

### Q-Learning Agent

The agent lies at the heart of the implements the Q-Learning algorithm by using the Deep Q-Network as a function approximator that gradually learns how to accurately predict the Q-value for every state+action combination, thereby allowing the agent to follow the action that offers the best expected reward.

The following optimizations have been used:
- **`Experience replay`**: A replay buffer has been implemented that stores the last 'BUFFER_SIZE' experience tuples, with each tuple holding the appropriately named values: <State, Action, Reward, Next-State, Done>. Experience replay allows the agent to train its Q-Network using raw cause->effect data that prevents the agent from deriving any implicit correlations between pieces of chronological data. This is analogous to the concept of shuffling that exploits the stochasticity of mini-batches that are randomly selected from the training data, for training regular deep neural networks.
- **`Fixed Q-Targets`**: 2 Q-Networks are used by the agent internally - the local Q-Network and the target Q-Network. Periodically, the parameters from the local Q-Network are copied over (with a dampening factor 'TAU') to the target Q-Network.

The final algorithm, incorporating the aforementioned optimizations into the vanilla Q-learning algorithm is as follows:
![Q-Learning][image3]

The following hyperparameter values have been used for this algorithm:
- **`BUFFER_SIZE`** : int(1e5) # replay buffer size
- **`GAMMA`**: 0.99            # discount factor
- **`UPDATE_EVERY`** : 40      # steps after which to copy over parameters to the fixed Q-target 
- **`TAU`** : 1e-3             # for soft update of target parameters

### Deep Q-Network

Deep Learning has found tremendous use in reinforcement learning, because with sufficient depth, such a network can approximate _*any*_ arbitrary function, according to the 'Universal Approximation Theorem'. The solution to a reinforcement learning problem too can be viewed as a multivariate function with multiple inputs and multiple outputs. Here, the inputs comprise the 'state' of the environment that is relevant to the agent and its actions; the outputs are the expected cumulative rewards that the agent can expect to receive (by the end of the game) for each possible action it takes from the given input state...i,e the Q-Value for the given state and each action.

Training the network requires letting the agent play a set of games, until its performance on the game is considered adequate. Various hyperparameters, as with any other deep learning solution, are required to be set for this to be achieved with the fewest possible iterations (episodes) of the game. These hyperparameters will be discussed in the sections to follow, as needed.

#### Network Architecture

Here is the architecture of the network that is used by the agent:
```
QNetwork(
  (fc1): Linear(in_features=37, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=64, bias=True)
  (fc4): Linear(in_features=64, out_features=4, bias=True)
)
```

The following hyperparameter values have been used:
- **`BATCH_SIZE`** : 128       # batch size
- **`LR`** : 5e-4              # learning rate 


## Instructions

#### Training the agent

Open ` ` and 

#### Running the agent

Open ` ` and

## Results

There are 3 graphs here.

1. The first one shows the "Accuracy" with which the agent picks bananas. It is the ratio of the yellow bananas picked to the total bananas picked. In essence, it conveys how efficiently the agent picked the yellow bananas while _avoiding_ the blue bananas. Any value less than 1 just means that the agent picked up some blue bananas along the way. Obviously, the accuracy should be as close to 1 as possible.
2. The second graph shows the actual total score per episode. Assuming a high accuracy score, this value also points to how efficiently the agent navigated the _open spaces_ to pick up as many bananas it could.
3. The third graph is what this project is actually after. It shows the average score over the last 100 episodes, or all prior episodes, whichever is smaller. As you can see, the agent is able to finally settle on a score of ~20 over a 100-episode period, after ~2500-3000 episodes of training. It should be noted that the agent actualy hit the 13+ mark around the 1500th episode itself!

![Performance Graph][image2]


## Future Enhancements

One of the drawbacks of the implementation above is that the agent took a long time (~1500 episodes) to reach the 100-episode average score of 13+. This can be improved in several ways. Here are a few such techniques.

1. Incorporating Double DQN Learning
2. Incorporating Duelling Q-Learning
3. Prioritized experience replay
4. Learning from Pixels (which is discussed at length in the appendix)

## Appendix

### Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, another Unity environment needs to be downloaded. This environment is almost identical to the previous environment, except that the state retrieved from the environment is a 84 x 84 RGB image, corresponding to the agent's first-person view.

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder and unzip (or decompress) the file. Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
