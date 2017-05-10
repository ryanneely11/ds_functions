
##HMM model
##Functions/class to implement an actor that 
##uses a HMM model to make action choices

import numpy as np

"""
A class to implement an HMM-based agent
Inputs:
	-alpha: equivalence point between action values
	-beta: the inverse temperature (explore/exploit param)
	-delta: the transition probability
	-correct_mean: probability of recieving a reward given the correct
		choice (estimated from subjects' behavior; same as expected value)
	-incorrect_mean: probability of receiving a reward given the incorrect
		choice (estimated)
	-bandit: a bandit class to interact with the agent
	-actions: a list of actions to take (can be floats). 
		 **restricted in this model to two total values**
"""
class HMM_agent(object):
	def __init__(self,bandit,actions,alpha,beta,delta,
		correct_mean,incorrect_mean):
		self.bandit = bandit
		self.actions = actions
		self.alpha = alpha
		self.beta = beta
		self.delta = delta
		self.correct_mean = correct_mean
		self.incorrect_mean = incorrect_mean
		self.states = [1,2]
		self.log = {
		'PX_correct':[0.5], ##Probability distribution over states
		'outcome':[0],##reward history
		'action':[actions[0]], ##action history
		'p_switch':[0.5], ##probability of switching
		'p_a':[0.5], ##probability of action a
		'p_b':[0.5] ##probability of action b
		}

	##run one trial and update based on the Markov Property
	##and Bayes rule
	def run(self):
		P_switch = self.compute_p_switch()
		action = self.choose_action(P_switch)
		choice = self.was_switch(action) ##did we switch or stay?
		prior_Xt = self.compute_prior(choice)
		##compute the probabilities of taking each action
		p_a,p_b = self.get_action_probs(P_switch)
		##now take the action
		outcome = self.bandit.run(action)
		##now compute the posterior
		posterior_Xt = self.compute_posterior(outcome,prior_Xt)
		##now save everything to the log
		self.log['PX_correct'].append(posterior_Xt)
		self.log['outcome'].append(outcome)
		self.log['action'].append(action)
		self.log['p_switch'].append(P_switch)
		self.log['p_a'].append(p_a)
		self.log['p_b'].append(p_b)
	

	"""`
	Compute the probability of making an action switch
	Inputs:
		Xt_post: Posterior probability that X(t) is correct
		alpha: state equivalence parameter
		beta: inverse temp param (explore/exploit)
	Returns:
		P_switch: probability of making an action switch
	"""
	def compute_p_switch(self):
		P_incorrect = 1-self.log['PX_correct'][-1]##probability that the action is incorrect
		return self.sigmoid_choice(P_incorrect)

		"""
	A function to take an action based on the probability of
	making a switch
	Inputs:
		P_switch: switch probability
	Returns:
		action: the chosen action
	"""
	def choose_action(self,P_switch):
		choices = ['switch','stay']
		probs = [P_switch,1-P_switch]
		choice = np.random.choice(choices,p=probs)
		if choice == 'switch':
			##take the alternate action from the last trial
			action = [x for x in self.actions if x != self.log['action'][-1]][0]
		elif choice == 'stay':
			action = self.log['action'][-1]
		return action

	"""
	A function to compute the probabilities of action a and b
	Inputs:
		p_switch: the probability of switching on the current trial
	Returns:
		p_a: probability of choosing action a
		p_b: probability of choosing action b
	"""
	def get_action_probs(self,P_switch):
		last_action = self.log['action'][-1]
		if last_action == self.actions[0]:
			p_b = P_switch
			p_a = 1-P_switch
		elif last_action == self.actions[1]:
			p_a = P_switch
			p_b = 1-P_switch
		return p_a,p_b

	"""
	A function to determine if the current choice represents
	a switch from the previous choice
	Inputs:
		new_choice
	Returns: "switch" or "stay" depending on what the subject did
	"""
	def was_switch(self,new_choice):
		if new_choice == self.log['action'][-1]:
			result = 'stay'
		else:
			result = 'switch'
		return result 

	"""
	A function to compute the Bayesian prior based on the upcoming choice, the 
	posterior from the previous trial, and the transition probability.
	Input:
		choice; 'switch' or 'stay'
	Returns: 
		prior_Xt: estimate that the current state is correct
	"""
	def compute_prior(self,choice):
		##get the posterior from the previous trial
		posterior_Xt = self.log['PX_correct'][-1]
		##get the transition probabilities for the different states
		P_trans = self.compute_p_transition(choice) ##this is a list with 2 values
		return (P_trans[0]*posterior_Xt)+(P_trans[1]*(1-posterior_Xt))

	"""
	A function to compute the posterior given the outcome of the trial, 
	as well as the prior, and some info about the reward probabilities.
	Inputs:
		outcome: the outcome of the current trial
		prior_Xt: the prior estimate of the current trial
	Returns:
		posterior_Xt
	"""
	def compute_posterior(self,outcome,prior_Xt):
		##compute the probability of this outcome given different state possibilities
		P_outcomes = self.compute_p_outcome(outcome)
		posterior = (P_outcomes[0]*prior_Xt)/((P_outcomes[0]*prior_Xt)+(
			P_outcomes[1]*(1-prior_Xt)))
		#diagnostics:
		if posterior < 0 or posterior > 1:
			print("posterior error: prior={}, p_outcome[0]={}, p_outcome[1]={}".format(
				prior_Xt,P_outcomes[0],P_outcomes[1]))
		return posterior
	"""
	Returns the probability that the state is correct given the choice to stay or 
	switch, computed for (index[0]), the case where the previous state was correct
	and (index[1]) the case where the previous state was incorrect.
	Inputs:
		choice; either 'stay' or 'switch'
	Retuns:
		P_transition: list of of probabilities that the current state estimate is correct
	"""
	def compute_p_transition(self,choice):
		P_transition = [0,0]
		if choice == 'switch':
			P_transition[0] = self.delta ##case where the state was correct but we switched
			P_transition[1] = 1-self.delta ##case where the state was incorrect and we switched
		elif choice == 'stay':
			P_transition[0] = 1-self.delta
			P_transition[1] = self.delta
		return P_transition

	"""
	A function to compute the probability of recieving a reward given
	being in one of the states. 
	Inputs:
		reward value, either 1 or 0
	Returns:
		P_outcomes: the probability of receiving the given outcome
			assuming you were in [state1(correct), state2(incorrect)]
	"""
	def compute_p_outcome(self,outcome):
		P_outcomes = [0,0]
		if outcome == 1:
			P_outcomes[0] = np.random.binomial(10,self.correct_mean)/10
			P_outcomes[1] = 1-P_outcomes[0]
		elif outcome == 0:
			P_outcomes[0] = np.random.binomial(10,self.incorrect_mean)/10
			P_outcomes[1] = 1-P_outcomes[0]
		##make sure we don't have any zeros
		for i in range(len(P_outcomes)):
			if P_outcomes[i] == 0:
				P_outcomes[i] = 0.02
		return P_outcomes

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
	An alternate sigmoidal choice function
	"""
	def sigmoid_choice(self,z):
		return 1.0/(1+np.exp(-self.beta*(z-self.alpha)))


