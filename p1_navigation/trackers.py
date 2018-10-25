import numpy as np
import matplotlib
import matplotlib.pyplot as plt

AVERAGING_WINDOW = 100

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
        self.running_averages = np.zeros(num_episodes, dtype=np.float32)
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
            self.accuracies[episode] = (1.0 * self.positive_counts[episode]) / (self.positive_counts[episode] + 
                                                                                self.negative_counts[episode])

            if episode >= AVERAGING_WINDOW:
                self.running_averages[episode] = np.mean(self.scores[episode-AVERAGING_WINDOW:episode])
            else:
                self.running_averages[episode] = np.mean(self.scores[0:episode])

    def plot(self):
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Accuracy", episodes, "Episode", self.accuracies, "Accuracy", id=311) 
        self.__plot__("Scores", episodes, "Episode", self.scores, "Score", id=312) 
        self.__plot__("{}-Episode Running Averages".format(AVERAGING_WINDOW), episodes, "Episode", self.running_averages, "{}-Episode Average Score".format(AVERAGING_WINDOW), id=313)
        plt.show()
            
    def plot_accuracies(self):
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Accuracy", episodes, "Episode", self.accuracies, "Accuracy") 
        plt.show()
    
    def plot_scores(self):
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Scores", episodes, "Episode", self.scores, "Score") 
        plt.show()

    def plot_running_averages(self):
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("{}-Episode Running Averages".format(AVERAGING_WINDOW), episodes, "Episode", self.running_averages, "{}-Episode Average Score".format(AVERAGING_WINDOW))
        plt.show()

    def __plot__(self, title, xvalues, xlabel, yvalues, ylabel, id=111):
        fig = plt.figure()
        ax = fig.add_subplot(id)
        ax.plot(xvalues, yvalues)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
#         ax.grid()
#         fig.savefig("test.png")
