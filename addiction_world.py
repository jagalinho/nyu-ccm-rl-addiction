class AddictionWorld():
    def __init__(self, max_time, n_trials, rewards=[], verbose=False):
        '''
        rewrads -> list of (reward, time, (first_trial, last_trial))
        '''
        self.max_time = max_time
        self.n_trials = n_trials
        self.reward = np.zeros((self.n_trials, self.max_time))

        for r, t, trials in rewards:
            self.add_reward(r, t, trials, verbose=verbose)

    def add_reward(self, reward, time, trials, verbose=False):
        if time < 1 or time > self.max_time:
            print("Tried to add reward on time {} when max_time is {}"
                .format(time, self.max_time))
            return False
        
        if trials[0] < 1 or trials[1] > self.n_trials:
            print("Tried to add rewards for trials {}-{} when there are only trials 1-{}"
                .format(trials[0], trials[1], self.n_trials))
            return False
        
        if verbose:
            print("Adding reward of {} on time {} for trials {}-{}"
                .format(reward, time, trials[0], trials[0]))
        for i in range(trials[0]-1, trials[1]):
            if verbose and self.world[i][time-1] != 0:
                print("Replacing current reward of {} on time {} for trial {} with new reward of {}"
                    .format(self.world[i][time-1], time, i, reward))
            self.reward[i][time-1] = reward
        return True

    def clear_reward(self, time, trials, verbose=False):
        if time < 1 or time > self.max_time:
            print("Tried to clear rewards on time {} when max_time is {}"
                .format(time, self.max_time))
            return False

        if trials[0] < 1 or trials[1] > self.n_trials:
            print("Tried to clear rewards for trials {}-{} when there are only trials 1-{}"
                .format(trials[0], trials[1], self.n_trials))
            return False
        
        if verbose:
            print("Clearing rewards on time {} for trials {}-{}"
                .format(time, trials[0], trials[0]))
        for i in range(trials[0]-1, trials[1]):
            self.reward[i][time-1] = 0
        return True

    def reset(self, max_time=-1, n_trials=-1, rewards=[], verbose=False):
        max_time = self.max_time if max_time == -1 else max_time
        n_trials = self.n_trials if n_trials == -1 else n_trials

        if verbose:
            print("Reseting world with max_time as {} and {} trials"
                .format(max_time, n_trials))
        self.reward = np.zeros((n_trials, max_time))

        for r, t, trials in rewards:
            self.add_reward(r, t, trials, verbose=verbose)
    
    def run_TD(self, ALPHA=1, GAMMA=1):
        """
        Function which simulates trials, updating values and recording prediction error
        
        ALPHA -> How important actual reward is when updating values (as opposed to previous value)
        GAMMA -> How important future expected reward is (discount factor)
        """
        value = np.zeros((self.n_trials+1, self.max_time))
        prediction_error = np.zeros(self.reward.shape)

        for trial in range(self.n_trials):
            for t in range(self.max_time):
                expected_value = value[trial][t]
                future_value = value[trial][t+1] if t+1 != self.max_time else 0
                actual_value = self.reward[trial][t] + (GAMMA * future_value)
                prediction_error[trial][t] = actual_value - expected_value
                # Based on the Princeton paper, uses ALPHA as an importance weight for actual reward vs expectation
                value[trial+1][t] = ((1 - ALPHA) * expected_value) + (ALPHA * actual_value)

        return value[1:], prediction_error