[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
![Trained Agent][image1]

## Overview

This project aims to train a Deep Learning Agent to navigate (and collect bananas!) in a large, square world. 

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

## Getting Started

1. The game runs in a Unity environment, which can be downloaded from one of the links below, according to the operating system where this is to be run:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, under the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Dependencies

A python 3.5+ environment with PyTorch version 0.4.0+ must be available for the agent to run. Additionally, it would be advisable if the system can make use of a GPU. If using a NVidia GPU, it is recommended to use the latest NVidia GPU driver (4.10 as of this writing).

## Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

## Implementation


## Future Enhancements

The following are a few enhancements that can be made for the agent to learn faster and to improve its efficiency at collecting yellow bananas and avoiding blue bananas.

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
