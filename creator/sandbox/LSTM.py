import numpy as np


#code derived from Karpathy's gist
#enhanced with guidance from Graves' book:
#	www.cs.toronto.edu/~graves/preprint.pdf


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
		# model parameters for the LSTM layer
		self.W_i = np.random.randn(self.hidden_size, \
			self.hidden_size + self.vocab_size)*0.01 # inputs
		self.W_f = np.random.randn(self.hidden_size, \
			self.hidden_size + self.vocab_size)*0.01 # forget cells
		self.W_c = np.random.randn(self.hidden_size, \
			self.hidden_size + self.vocab_size)*0.01 # update cells
		self.W_o = np.random.randn(self.hidden_size, \
			self.hidden_size + self.vocab_size)*0.01 # output 
		

		#output layer
		self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # output 
		self.mWhy = np.zeros_like(self.Why)
		self.by = np.zeros((self.vocab_size, 1))
		self.mby = np.zeros_like(self.by)

		self.b_i = np.zeros((self.hidden_size, 1)) # input bias
		self.b_f = np.zeros((self.hidden_size, 1)) # forget bias
		self.b_c = np.zeros((self.hidden_size, 1)) # cell state bias
		self.b_o = np.zeros((self.hidden_size, 1)) # output bias
	
		#self.n, self.p = 0, 0
		self.mW_i =  np.zeros_like(self.W_i)
		self.mW_f =  np.zeros_like(self.W_f)
		self.mW_c =  np.zeros_like(self.W_c)
		self.mW_o =  np.zeros_like(self.W_o)
		
		self.mb_i = np.zeros_like(self.b_i)
		self.mb_f = np.zeros_like(self.b_f)
		self.mb_c = np.zeros_like(self.b_c)
		self.mb_o = np.zeros_like(self.b_o)
	
		self.C = np.random.randn(self.hidden_size,1)*0.01
		self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0

	def activation_function(self, x, h):
		return self.LSTM(x,h)

	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
	
	def d_sigmoid(self, x):
		return self.sigmoid(x)*(1-self.sigmoid(x))

	def lossFun(self, inputs, targets, hprev, cprev):
		"""
		inputs,targets are both list of integers.
		hprev is Hx1 array of initial hidden state
		returns the loss, gradients on model parameters, and last hidden state
		"""
		xs, hs, ys, ps = {}, {}, {}, {}

		f, i, C_prime, o, C, h = {}, {}, {}, {}, {}, {}
		C[-1] = np.copy(cprev)		
		h[-1] = np.copy(hprev)
		loss = 0
		
		for t in range(len(inputs)):
			xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
			xs[t][inputs[t]] = 1

			h_x = np.concatenate([h[t-1],xs[t]]) # Ax + By = [A|B][x/y] 
			i[t] = self.sigmoid(np.dot(self.W_i, h_x) + self.b_i) # input gate layer
			f[t] = self.sigmoid(np.dot(self.W_f, h_x) + self.b_f) # forget gate layer
			o[t] = self.sigmoid(np.dot(self.W_o, h_x) + self.b_o) # output layer
			C_prime[t] = np.tanh(np.dot(self.W_c, h_x) + self.b_c) #candidate values

			C[t] = np.multiply(f[t], C[t-1]) + np.multiply(i[t], C_prime[t])
			h[t] = o[t] * np.tanh(C[t]) # merge outputs with cell state
			
			ys[t] = np.dot(self.Why, h[t]) + self.by # actual output layer
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
			
			assert ps[t].shape == xs[t].shape,\
				(ps[t].shape, " should be equal in length to ",xs[t].shape) 
			loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

		#prep gradient stuff
		dW_i = np.zeros_like(self.W_i)
		dW_c = np.zeros_like(self.W_c)
		dW_f = np.zeros_like(self.W_f)
		dW_o = np.zeros_like(self.W_o)
		dWhy  = np.zeros_like(self.Why)
		db_i, db_c,  = np.zeros_like(self.b_i), np.zeros_like(self.b_c)
		db_f, db_o,  = np.zeros_like(self.b_f), np.zeros_like(self.b_o)
		dby = np.zeros_like(self.by)
		dhnext = np.zeros_like(h[0])
		dcnext = np.zeros_like(h[0])
		# backward pass: compute gradients going backwards
		for t in reversed(range(len(inputs))):

			#start at the outcomes
			dy = np.copy(ps[t])
			dy[targets[t]] -= 1 # backprop into output
			dWhy += np.dot(dy, h[t].T) # modify weights for outputs
			dby +=dy # modify bias

			#how have to backprop out through the LSTM
			dh = np.dot(self.Why.T, dy) + dhnext #backprop into h
			
			dC = o[t] * dh *  dcnext 
			do = C[t] * dh
			di = C_prime[t] * dC
			dc = i[t] * dC
			df = self.C[t-1] * dC

			di_input = (1.0 - i[t]) * i[t] * di
			df_input = (1.0 - f[t]) * f[t] * df
			do_input = (1.0 - o[t]) * o[t] * do
			dc_input = (1.0 - C_prime[t]) * C_prime[t] * dc

			dW_i += np.outer(di_input.T, h_x)
			dW_f += np.outer(df_input.T, h_x)
			dW_o += np.outer(do_input.T, h_x)
			dW_c += np.outer(dc_input.T, h_x)

			db_i += di_input
			db_f += df_input
			db_o += do_input
			db_c += dc_input

		for dparam in [dWhy, dW_i, dW_c, dW_f, dW_o, dby, db_i, db_c, db_f, db_o]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return loss, dWhy, dW_i, dW_c, dW_f, dW_o, dby, db_i, db_c, db_f, db_o, h[len(inputs)-1]
	
	
	#def sample(self, h, seed_ix, n):
	#	""" 
	#	sample a sequence of integers from the model 
	#	h is memory state, seed_ix is seed letter for first time step
	#	"""
	#	x = np.zeros((self.vocab_size, 1))
	#	x[seed_ix] = 1
	#	ixes = []
	#	for t in range(n):
	
	#		'''
	#		hs[t] = self.activation_function(xs[t], hs[t-1])
	#		#this is in the activation function now?
	#		ys[t] = np.dot(self.Why, hs[t]) + self.by 
	#		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
	#		loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
	#		'''

	#		#h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
	#		h = self.activation_function(x, h)

	#		print(self.W_o.shape)
	#		print(h.shape)
	#		y = np.dot(self.W_o, h) + self.b_o
	#		p = np.exp(y) / np.sum(np.exp(y))
	#		ix = np.random.choice(range(self.vocab_size), p=p.ravel())
	#		x = np.zeros((self.vocab_size, 1))
	#		x[ix] = 1
	#		ixes.append(ix)
	#	return ixes

	def train(self, iterations):
		n, p = 0, 0
		while n < iterations:
			# prepare inputs (we're sweeping from left to right in steps seq_length long)
			if p+self.seq_length+1 >= len(self.data) or n == 0: 
				hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
				cprev = np.zeros((self.hidden_size,1)) # resent cell state
				p = 0 # go from start of data
			inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length]]
			targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1]]

			# sample from the model now and then
			#if n % 1000 == 0:
			#	#this is our get function
			#	sample_ix = self.sample(hprev, inputs[0], 200)
			#	txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
			#	print('----\n %s \n----' % (txt, ))

			# forward seq_length characters through the net and fetch gradient
			loss, dWhy, dW_i, dW_c, dW_f, dW_o, dby, db_i, db_c, db_f, db_o, hprev = \
				self.lossFun(inputs, targets, hprev, cprev)
			self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
			if n % 100 == 0: print('iter %d, loss: %f' % (n, self.smooth_loss)) # print progress

			# perform parameter update with Adagrad
			for param, dparam, mem in zip([self.Why, self.W_i, self.W_c, self.W_f, self.W_o, \
										   self.by, self.b_i, self.b_c, self.b_f, self.b_o], 
										  [dWhy, dW_i, dW_c, dW_f, dW_o, \
										   dby, db_i, db_c, db_f, db_o], 
										  [self.mWhy, self.mW_i, self.mW_c, self.mW_f, self.mW_o, \
										   self.mby, self.mb_i, self.mb_c, self.mb_f, self.mb_f]):
				mem += dparam * dparam
				param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

			p += self.seq_length # move data pointer
			n += 1 # iteration counter 

if __name__ == '__main__':
	RNN = RNN('input/case_sample.xml')
	RNN.train(10000)

