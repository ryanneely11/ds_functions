##RL_model: a reinforecement learning model
##implemented using the Rescorla Wagner rule

import numpy as np

"""
A class to implement a reinforcement learning
agent, using the Rescorla-Wagner rule.
Parameters (to be fitted):
	-alpha: equivalence point between action values
	-beta: the inverse temperature (explore/exploit param)
	-eta: the learning rate
	-bandit: a bandit class to interact with the agent
	-actions: a list of actions to take (can be floats). 
		 **restricted in this model to two total values**
"""
class RL_agent(object):
	def __init__(self,bandit,actions,alpha,beta,eta):
		self.alpha = alpha
		self.beta = beta
		self.eta = eta
		self.bandit = bandit
		self.actions = actions
		self.log = {
		'Qa':[0.5], ##action values of action a, updated each trial
		'Qb':[0.5], ##action value of action b
		'reward':[0], ##reward history
		'action':[actions[0]], ##action history
		'p_switch':[0.5] ##switch probability for comparing to state-based model
		}

	##run one trial
	def run(self):
		##if this is the first trial, seed the initial action values
		Qa_last = self.log['Qa'][-1] ##the last predicted action value
		Qb_last = self.log['Qb'][-1]
		reward_last = self.log['reward'][-1]
		action_last = self.log['action'][-1]
		##we only update the action value of the action that we had experience 
		##with on the last trial
		if action_last == self.actions[0]: ##case where we should update Qa
			d = self.get_delta(reward_last,Qa_last)
			Qa_prior = self.predictQ(Qa_last,self.eta,d)
			Qb_prior = Qb_last
		elif action_last == self.actions[1]: ##case where we should update Qb
			d = self.get_delta(reward_last,Qb_last)
			Qa_prior = Qa_last
			Qb_prior = self.predictQ(Qb_last,self.eta,d)
		else:
			raise ValueError
			print("Unknown action type: {}".format(action_last))
		##now we can predict what action we will take
		action,p_switch = self.choose_action(Qa_prior,Qb_prior,self.beta,self.alpha)
		##now feed this action into the bandit to get a reward
		reward = self.bandit.run(action)
		##finally, we can log all of the events from this action
		self.log['Qa'].append(Qa_prior)
		self.log['Qb'].append(Qb_prior)
		self.log['reward'].append(reward)
		self.log['action'].append(action)
		self.log['p_switch'].append(p_switch)


	"""
	A function to return an action value based on
	the value of that action at the previous time-step,
	some learning rate parameter, and a delta value, which
	is the difference between expected and actual reward
	Inputs:
		Qpost: last estimate of action value
		n: learning rate
		delta: difference between expected and received reward
			in the previous trial
	Returns:
		Qprior: predicted action value
	"""
	def predictQ(self,Qpost,eta,delta):
		return Qpost + (eta*delta)

	"""
	A function to compute the difference between extected
	and actual reward (delta)
	Inputs:
		r_actual: value of received reward
		r_predicted: predicted reward value
	Returns
		delta (difference)
	"""
	def get_delta(self,r_actual,r_predicted):
		return r_actual-r_predicted


	"""
	Compute the probability of choosing 
	action a (Pa), then make a choice based on this probability.
	Inputs:
		Qa: value of choice a
		Qb: value of choice b
		beta: inverse temperature parameter
		alpha: indecision point
	Returns:
		action: chosen action
	"""
	def choose_action(self,Qa,Qb,beta,alpha):
		##probability of choosing action a
		Pa = self.luce_choice(beta*((Qa-Qb)-alpha))
		p_switch = self.get_p_switch(Pa)
		##probability of choosing either action
		probs = [Pa,1-Pa]
		return np.random.choice(self.actions,p=probs),p_switch

	"""
	A helper function to imlement the 
	Luce Choice Rule.
	Inputs:
		z: value to use in computation
	returns:
		Pa: action probability
	"""
	def luce_choice(self,z):
		return 1.0/(1+np.exp(-z))

	"""
	A function to determine the probability of switching actions,
	given the probability of choosing action A.
	Inputs:
		Pa: probability of choosing action A
	Returns:
		p_switch: probability of switching actions
	"""
	def get_p_switch(self,Pa):
		##determine what the last action was
		if self.log['action'][-1] == self.actions[0]:
			##case where the last action was action A
			p_switch = 1-Pa ##probability of switchin is prob of action B
		elif self.log['action'][-1] == self.actions[1]:
			##case where last action was action B
			p_switch = Pa
		return p_switch





