##bandit task: a class to simulate the task
##and interact with action-learner models

import numpy as np

"""
A class to simulate the reversal learning task, based on
empirical paramsters that can be set externally.
Inputs:
	actions: a list of (2) actions to choose from
	p_rewarded: the probability of a correct action being rewarded
	switch_after: the number of trials to switch after (a list!) (plus some jitter)
"""
class bandit(object):
	def __init__(self,actions,p_rewarded,switch_after):
		self.p_rewarded = p_rewarded
		self.actions = actions
		self.switch_after = switch_after
		##randomly select one lever to be the first rewarded option
		self.correct_state = actions[0]
		##possible outcomes
		self.outcomes = [1,0]
		##corresponding probabilities
		self.probs = [p_rewarded,1-p_rewarded]
		##keep track of how many trials we've done
		self.n_trials = 0
	"""
	A function that runs one trial.
	Input:
		-action: input action
	Returns:
		-outcome
	"""
	def run(self,action):
		##see if a reversal occurred
		self.check_reversal()
		##now see if this action was rewarded
		if action == self.correct_state:
			outcome = np.random.choice(self.outcomes,p=self.probs)
		else:
			outcome = self.outcomes[1]
		self.n_trials +=1
		return outcome

	"""
	A function to reverse the context
	"""
	def check_reversal(self):
		if self.n_trials in self.switch_after:
			self.correct_state = [x for x in self.actions if not x == self.correct_state][0]