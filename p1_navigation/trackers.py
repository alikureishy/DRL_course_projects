import numpy as np

def class BananaTracker(object):
    
    def __init__(self, num_episodes):
        self.scores = np.zeros(num_episodes, dtype=np.int32).
        self.positives = np.zeros(num_episodes, dtype=np.int32)
        self.positive_counts = np.zeros(num_episodes, dtype=np.int32)
        self.negatives = np.zeros(num_episodes, dtype=np.int32)
        self.negative_counts = np.zeros(num_episodes, dtype=np.int32)
        self.neutral_counts = np.zeros(num_episodes, dtype=np.float32)
        self.steps = np.zeros(num_episodes, dtype=np.int32)
        self.accuracies = np.zeros(num_episodes, dtype=np.float32)
        self.recalls = np.zeros(num_episodes, dtype=np.float32)
        self.num_episodes = num_episodes
        
    def step(episode, reward, done):
        assert episode >= 0 and episode < self.num_episodes, "Invalid episode number: {}".format(episode)

        self.steps[episode] += 1
        if reward < 0:
            self.negatives[episode] += reward
            self.negatives_counts[episode] += 1
        elif reward == 0:
            self.neutral_counts[episode] += 1
        elif reward > 0:
            self.positives[episode] += reward
            self.positive_counts[episode] += 1
            
        if done:
            self.scores[episode] = self.positives[episode] + self.negatives[episode]
            self.accuracies[episode] = 1.0 * self.positive_counts[episode] / self.steps[episode]
            
    def plot_steps(self, for_episodes=None):
        pass
    
    def plot_accuracies(self, for_episodes=None):
        pass
    
    def plot_scores(self, for_episodes=None):
        pass