#!/usr/bin/env python

"""
Agent class for addiction RL modeling

authors: kpant, jagalinho
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt


class AddictionAgent:
    """
    Addiction agent class, used in RL
    """
    def __init__(self, max_time, n_trials, ETA=1, GAMMA=1, rewards=[], substances=[], verbose=False):
        """
        Initialize agent
        :param max_time: Number of time states in a trial
        :param n_trials: Number of trials to run
        :param ETA: Learning rate for TD learning
        :param GAMMA: Discount rate for future rewards
        :param rewards: List of (reward_size, time_state, (first_trial, last_trial)) tuples
        :param substances: List of (reward_size, addiction_size, time_state, (first_trial, last_trial)) tuples
        """
        self.max_time = max_time
        self.n_trials = n_trials
        self.reward = np.zeros((self.n_trials, self.max_time))
        self.addiction = np.zeros((self.n_trials, self.max_time))
        self.value = np.zeros((self.n_trials, self.max_time))
        self.prediction_error = np.zeros((self.n_trials, self.max_time))
        self.learning_rate = ETA
        self.discount_rate = GAMMA

        for args in rewards:
            self.add_reward(*args, verbose=verbose)

        for args in substances:
            self.add_substance(*args, verbose=verbose)

    def add_reward(self, reward, time, trials=None, verbose=False):
        """
        Update reward table with specified reward details
        :param reward: size of reward
        :param time: time step to update
        :param trials: Tuple of (first_trial, last_trial), if None update all trials
        :param verbose: Include more print statements (for debugging)
        :return: self
        """
        if time >= self.max_time:
            raise ValueError(f"Tried to add reward on time {time} when max_time is {self.max_time}")

        if trials is None:
            trial_range = (0, self.n_trials)
        else:
            if trials[1] >= self.n_trials:
                raise ValueError(
                    f"Tried to add rewards for trials {trials[0]}-{trials[1]} when there are only {self.n_trials}")
            trial_range = trials
        
        if verbose:
            print(f"Adding reward of {reward} on time {time} for trials {trial_range}")
        
        self.reward[trial_range[0]:trial_range[1], time] = reward

        if verbose:
            print('Reward added')
            print(self.reward)
        
        return self

    def add_substance(self, reward, addiction, time, trials=None, verbose=False):
        """
        Add an addicting substance to reward and substance tables
        :param reward: The non-addicting reward component size
        :param addiction: The addicting component size
        :param time: Time step to update
        :param trials: Tuple of (first_trial, last_trial), if None update all trials
        :param verbose: Include more print statements (for debugging)
        :return: self
        """
        self.add_reward(reward, time, trials, verbose)

        if trials is None:
            trial_range = (0, self.n_trials)
        else:
            trial_range = trials

        if verbose:
            print(f"Adding addiction of {addiction} on time {time} for trials {trial_range}")
        
        self.addiction[trial_range[0]:trial_range[1], time] = addiction

        if verbose:
            print('Substance added')
            print(self.addiction)
        
        return self

    def clear_reward(self, time, trials, verbose=False):
        """
        Clear reward from a specified time and trial set
        :param time: Time step to clear reward
        :param trials: Tuple of (first_trial, last_trial), if None update all trials
        :param verbose: Include more print statements (for debugging)
        :return: self
        """
        if time >= self.max_time:
            raise ValueError(f"Tried to clear rewards on time {time} when max_time is {self.max_time}")

        if trials[1] >= self.n_trials:
            raise ValueError(
                f"Tried to clear rewards for trials {trials[0]}-{trials[1]} when there are only {self.n_trials}")

        if verbose:
            print(f"Clearing rewards on time {time} for trials {trials[0]}-{trials[1]}")
        
        self.reward[trials[0]:trials[1], time] = 0

        if verbose:
            print('Reward cleared')
            print(self.reward)
        
        return self

    def clear_substance(self, time, trials, verbose=False):
        """
        Clear addictive substance from substance table. Note this does not clear the reward at the specified states, to
        allow for more flexibility.
        :param time: Time step to clear substance
        :param trials: Tuple of (first_trial, last_trial), if None update all trials
        :param verbose: Include more print statements (for debugging)
        :return: self
        """
        if time >= self.max_time:
            raise ValueError(f"Tried to clear rewards on time {time} when max_time is {self.max_time}")

        if trials[1] >= self.n_trials:
            raise ValueError(
                f"Tried to clear rewards for trials {trials[0]}-{trials[1]} when there are only {self.n_trials}")

        if verbose:
            print(f"Clearing rewards on time {time} for trials {trials[0]}-{trials[1]}")
        
        self.addiction[trials[0]:trials[1], time] = 0

        if verbose:
            print('Addictive substance cleared')
            print(self.addiction)
        
        return self

    def run_TD(self, verbose=False):
        """
        Simulate temporal difference learning, update value and prediction error tables.
        :return: self
        """
        for trial in range(self.n_trials):
            for t in range(self.max_time - 1):
                expected_value = self.value[trial][t]
                actual_value = self.reward[trial][t] + self.discount_rate * (self.value[trial][t + 1])
                prediction_error = np.max([(actual_value - expected_value + self.addiction[trial][t]),
                                           self.addiction[trial][t]])
                self.prediction_error[trial][t] = prediction_error
                if trial < (self.n_trials - 1):
                    self.value[trial + 1][t] = expected_value + (self.learning_rate * prediction_error)
        
        if verbose:
            print('TD learning complete')
            print('Value Table:')
            print(self.value)
            print('Prediction Errors:')
            print(self.prediction_error)
        
        return self

    def __plot(self, fig, data, label, title):
        """
        Generate a 3D plot from given 2D data
        :param fig: pyplot Figure
        :param data: 2D numpy matrix
        :param label: String to label the data
        :param title: String to title the plot
        :return: pyplot Figure with plot added
        """
        X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((np.ptp(X), np.ptp(Y)/2, np.ptp(data)))
        ax.set_xlabel('Time')
        ax.set_ylabel('Trials')
        ax.set_zlabel(label)
        ax.set_title(title)

        ax.plot_surface(X, Y, data)

        return fig
    
    def plot_value(self, title=''):
        """
        Plot value table
        :param title: String to title the plot
        :return: self
        """
        fig = plt.figure()
        self.__plot(fig, self.value, 'Value', title)
        plt.show()

        return self

    def plot_prediction_error(self, title=''):
        """
        Plot prediction error
        :param title: String to title the plot
        :return: self
        """
        fig = plt.figure()
        self.__plot(fig, self.prediction_error, 'Prediction Error', title)
        plt.show()
        
        return self