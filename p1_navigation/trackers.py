import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class BananaTracker(object):
    
    def __init__(self, num_episodes):
        self.scores = np.zeros(num_episodes, dtype=np.int32)
        self.positives = np.zeros(num_episodes, dtype=np.int32)
        self.positive_counts = np.zeros(num_episodes, dtype=np.int32)
        self.negatives = np.zeros(num_episodes, dtype=np.int32)
        self.negative_counts = np.zeros(num_episodes, dtype=np.int32)
        self.neutral_counts = np.zeros(num_episodes, dtype=np.float32)
        self.step_counts = np.zeros(num_episodes, dtype=np.int32)
        self.accuracies = np.zeros(num_episodes, dtype=np.float32)
        self.recalls = np.zeros(num_episodes, dtype=np.float32)
        self.num_episodes = num_episodes
        
    def step(self, episode, reward, done):
        assert episode >= 0 and episode < self.num_episodes, "Invalid episode number: {}".format(episode)

        self.step_counts[episode] += 1
        if reward < 0:
            self.negatives[episode] += reward
            self.negative_counts[episode] += 1
        elif reward == 0:
            self.neutral_counts[episode] += 1
        elif reward > 0:
            self.positives[episode] += reward
            self.positive_counts[episode] += 1
            
        if done:
            self.scores[episode] = self.positives[episode] + self.negatives[episode]
            self.accuracies[episode] = 1.0 * self.positive_counts[episode] / (self.positive_counts[episode] + self.negative_counts[episode])
            
    def plot_accuracies(self, for_episodes=None):
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Accuracy", episodes, "Episode", self.scores, "Accuracy") 
    
    def plot_scores(self, for_episodes=None):
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Scores", episodes, "Episode", self.scores, "Score") 

    def __plot__(self, title, xvalues, xlabel, yvalues, ylabel):
        fig, ax = plt.subplots()
        ax.plot(xvalues, yvalues)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid()
#         fig.savefig("test.png")
        plt.show()