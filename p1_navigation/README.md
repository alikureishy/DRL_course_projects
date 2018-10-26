[//]: # (Image References)

[image1]: https://github.com/safdark/DRL_course_projects/blob/master/p1_navigation/docs/screenshot_banana_bot.png "Click for video of trained agent"

# Project 1: Navigation
[![Click for video of trained agent][image1]](https://youtu.be/2LZEazw_taM)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Instructions](#instructions)
	- [Visualizing the trained agent](#visualizing-the-trained-agent)
	- [Training the agent](#training-the-agent)

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
>> python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Now, make sure that from the Jupyter Notebook, you select the above kernel by going to "Kernel --> Change Kernel --> ___"".

5. Additionally, it would be advisable if the system can make use of a GPU. If using a NVidia GPU, it is recommended to use the latest NVidia GPU driver (4.10 as of this writing).

## Instructions

#### Visualizing the trained agent

Open [Navigation-Trainer](https://github.com/safdark/DRL_course_projects/edit/master/p1_navigation/Navigation-Trainer.ipynb) and run all the cells. This will launch a Unity environment and use the trained agent to make the robot navigate the field picking up the yellow bananas while avoiding the blue ones. The score should fall anywhere between 14-20, using the checkpoint.pth file included with this project.

#### Training the agent

1. Delete the 'checkpoint.pth' file.
2. Open the [Navigation-Visualizer](https://github.com/safdark/DRL_course_projects/edit/master/p1_navigation/Navigation-Visualizer.ipynb) notebook and run all the cells. This will take a while, depending on the speed of the CPU and the availability of a GPU.
3. Go back to the [Navigation-Trainer](https://github.com/safdark/DRL_course_projects/edit/master/p1_navigation/Navigation-Trainer.ipynb) file to visualize the agent's performance.

