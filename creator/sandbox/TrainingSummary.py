import pickle


# create a sample text
# save the loss
# compute score
# correct sentence
# compute new score
# save score, new score, loss, sentence, new sentence
class TrainingSummary():
	def __init__(self, t, loss, smooth_loss, sample):
		self.t = t
		self.loss = loss
		self.smooth_loss = smooth_loss
		self.sample = sample
