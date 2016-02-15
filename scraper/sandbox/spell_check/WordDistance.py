# to install, use:
# sudo apt-get install python3-levenshtein

# source code: 
# https://github.com/ztane/python-Levenshtein

import Levenshtein
import re

class WordDistance():
	def __init__(self, filename=None):
		if filename:
			self.loadDictionary(filename)

	def loadDictionary(self, filename):
		self.filename = filename
		dic = open(filename)
		self.dictionary = re.findall('[a-z]+', dic.read().lower())
		#dic = open('../training_text/dictionary.txt')
		dic.close()

	def closestWord(self, word):
		return self.closestNWords(word, 1)[0][0]
	
	def closestNWords(self, word, N):
		best = {d:self.jaro(word, d) for d in self.dictionary}
		return sorted(list(best.items()), key=lambda x: x[1], reverse=True)[:N]

	def jaro(self, word1, word2):
		return Levenshtein.jaro(word1, word2)


'''
# similarity between two strings - based on minimal edit distance
ratio = Levenshtein.ratio("Hello world!", "Holly grail!")
print("ratio: %s"%ratio)

# distance - compute absolute Levenshtein distance of two strings
distance = Levenshtein.distance("Hello world!", "Holly grail!")
print("distance: %s"%distance)

# hamming distance between two words
hamming = Levenshtein.hamming("Hello world!", "Holly grail!")
print("hamming: %s"%hamming)

# jaro similarity metric
jaro = Levenshtein.jaro("Hello world!", "Holly grail!")
print("jaro: %s"%jaro)

# jaro-winkler metric - like jaro but more weight in prefix, because spelling mistakes
# are more likely to occur near end of word
jaro_winlker = Levenshtein.jaro_winkler("Hello world!", "Holly grail!")
print("jaro_winlker: %s"%jaro_winlker)

# median - finds the median word based on a list - shortest distance between the rest
#	greedy algorithm
median =  Levenshtein.median(["Hello world!", "Holly grail!", "Hilly girl"])
print("median: %s"%median)
'''
