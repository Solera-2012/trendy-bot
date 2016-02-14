import numpy as np

class RNN():
	def __init__(self, filename):
		self.load_data(filename)
		self.set_hyperparameters()
		self.set_model_parameters()

	def load_data(self, filename):
		# data I/O
		self.data = open(filename, 'r').read() # should be simple plain text file
		self.chars = list(set(self.data))
		self.data_size, self.vocab_size = len(self.data), len(self.chars)
		print('data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
		self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
		self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }

	def set_hyperparameters(self):
		# hyperparameters
		self.hidden_size = 100 # size of hidden layer of neurons
		self.seq_length = 25 # number of steps to unroll the RNN for
		self.learning_rate = 1e-1
	
	def set_model_parameters(self):
		# model parameters
		self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01 # input->hidden
		self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden->hidden
		self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # hidden->output
		self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
		self.by = np.zeros((self.vocab_size, 1)) # output bias
	
		#self.n, self.p = 0, 0
		self.mWxh = np.zeros_like(self.Wxh)
		self.mWhh =  np.zeros_like(self.Whh)
		self.mWhy =  np.zeros_like(self.Why)
		self.mbh = np.zeros_like(self.bh)
		self.mby = np.zeros_like(self.by) # memory variables for Adagrad
		self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0

	def activation_function(self, x, h):
		return vanilla_rnn(x,h)

	def vanilla_rnn(self, x, h):
		#returns hs[t] in loss function or h in sample
		i_t = np.dot(self.Wxh, x)
		f_t = np.dot(self.Whh, h)
		return np.tanh(i_t + f_t + self.bh)

	def lossFun(self, inputs, targets, hprev):
		"""
		inputs,targets are both list of integers.
		hprev is Hx1 array of initial hidden state
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)
		loss = 0
		
		for t in range(len(inputs)):
			xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
			xs[t][inputs[t]] = 1

			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
			# unnormalized log probabilities for next chars
			ys[t] = np.dot(self.Why, hs[t]) + self.by 

			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
			loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
			print("during training: ps: ", ps[t].shape)

		# backward pass: compute gradients going backwards
		dWxh = np.zeros_like(self.Wxh)
		dWhh = np.zeros_like(self.Whh)
		dWhy = np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[0])
		for t in reversed(range(len(inputs))):
			
			#start from the outcomes
			dy = np.copy(ps[t])
			dy[targets[t]] -= 1 # backprop into y
			dWhy += np.dot(dy, hs[t].T) # modify weights for outputs
			dby += dy # modify bias
			dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
			
			#derivative of tanh = 1 - tanh^2(x)
			dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)
		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
	
	def sample(self, h, seed_ix, n):
		""" 
		sample a sequence of integers from the model 
		h is memory state, seed_ix is seed letter for first time step
		"""
		x = np.zeros((self.vocab_size, 1))
		x[seed_ix] = 1
		ixes = []
		for t in range(n):
			#run forward n times
			h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
			y = np.dot(self.Why, h) + self.by
			p = np.exp(y) / np.sum(np.exp(y))
			
			print("during sample: ", p.shape)
			#choose a random letter
			ix = np.random.choice(range(self.vocab_size), p=p.ravel())
			#encode the letter
			x = np.zeros((self.vocab_size, 1))
			x[ix] = 1
			ixes.append(ix)
		return ixes

	def train(self, iterations):
		n, p = 0, 0
		while n < iterations:
			# prepare inputs (we're sweeping from left to right in steps seq_length long)
			if p+self.seq_length+1 >= len(self.data) or n == 0: 
				hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
				p = 0 # go from start of data
			inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length]]
			targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1]]

			# sample from the model now and then
			if n % 1000 == 0:
				#this is our get function
				sample_ix = self.sample(hprev, inputs[0], 200)
				txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
				print('----\n %s \n----' % (txt, ))

			# forward seq_length characters through the net and fetch gradient
			loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
			self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
			if n % 100 == 0: print('iter %d, loss: %f' % (n, self.smooth_loss)) # print progress

			# perform parameter update with Adagrad
			for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
										  [dWxh, dWhh, dWhy, dbh, dby], 
										  [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
				mem += dparam * dparam
				param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

			p += self.seq_length # move data pointer
			n += 1 # iteration counter 

if __name__ == '__main__':
	RNN = RNN('input/case_sample.xml')
	RNN.train(1)

