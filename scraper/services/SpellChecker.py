import re, collections

class SpellChecker():
	def __init__(self):
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
		dic = open('scraper/training_text/dictionary.txt')
		big = open('scraper/training_text/big.txt')
		corpus = open('scraper/training_text/corpus.txt')

		self.training_set = self.words(dic.read() + big.read())
		self.NWORDS = self.train(self.training_set, 1)
		self.NWORDS = self.train(self.words(corpus.read()), 1000, self.NWORDS)

		dic.close(), big.close(), corpus.close()

	def words(self, text): 
		return re.findall('[a-z]+', text.lower()) 

	def train(self, features, weight, model=None):
		if model == None:
			model = collections.defaultdict(lambda: 1)
		for f in features:
			model[f] += weight
		return model

	def edits1(self, word):
	   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
	   deletes    = [a + b[1:] for a, b in splits if b]
	   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
	   replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
	   inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
	   return set(deletes + transposes + replaces + inserts)

	def known_edits2(self, word):
		return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) \
			if e2 in self.NWORDS)

	def known(self, words): 
		return set(w for w in words if w in self.NWORDS)

	def correct(self, word):
		candidates = self.known([word]) or self.known(self.edits1(word)) or \
			self.known_edits2(word) or [word]
		l = [self.NWORDS.get(c) for c in candidates]
		return max(candidates, key=self.NWORDS.get)
