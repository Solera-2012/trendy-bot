import pickle
from WordDistance import WordDistance 

class TrainingSummary():
	def __init__(self, t, loss, smooth_loss, sample):
		self.t = t
		self.loss = loss
		self.smooth_loss = smooth_loss
		self.sample = sample

	def __repr__(self):
		return "N:             %s\n\
				Loss:          %s\n\
				Smoothed loss: %s\n\
				Sample:        %s"%\
				(self.t, self.loss, self.smooth_loss, self.sample) 

	def addStats(self):
		word_fixer = WordDistance()
		self.score = word_fixer.sentenceScore(self.sample)
		self.sample_fixed = word_fixer.closestSentence(self.sample)
		self.score_fixed = word_fixer.sentenceScore(self.sample_fixed)
		return self


	def computeScore(self):
		pass

	def correctSentence(self):
		pass
	

# compute score
# correct sentence
# compute new score
