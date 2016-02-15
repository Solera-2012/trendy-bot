import pickle


class TrainingSummary():
	def __init__(self, t, loss, smooth_loss, sample):
		self.t = t
		self.loss = loss
		self.smooth_loss = smooth_loss
		self.sample = sample

	def __repr__(self):
		return "N:             %s\nLoss:          %s\nSmoothed loss: %s\nSample:        %s"%(self.t, self.loss, self.smooth_loss, self.sample) 

# compute score
# correct sentence
# compute new score
