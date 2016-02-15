import WordDistance 
import unittest
import random, string


class TestWordDistance(unittest.TestCase):
	def setUp(self):
		self.wordDistance = WordDistance.WordDistance('../training_text/dictionary.txt')

	def test_LoadDictionary(self):
		self.assertEqual(len(self.wordDistance.dictionary), 127142)

	def test_closestWord(self):
		close = self.wordDistance.closestWord("cowll")
		self.assertEqual("cowl", close)
		closer = self.wordDistance.closestWord("carv")
		self.assertEqual("carve", closer)

	def test_closetNWords(self):
		close = self.wordDistance.closestNWords("cowll",10)
		self.assertEqual(len(close), 10)

	def test_closestNWords_descending_order(self):
		words, probs = zip(*self.wordDistance.closestNWords("cowll",10))
		self.assertEqual(list(probs), sorted(probs,reverse=True))

	def test_correctSentenceScore1(self):
		score = self.wordDistance.sentenceScore("this is correct")
		self.assertEqual(score, 1)
	
	def test_incorrectSentenceScoreIsNot1(self):
		score = self.wordDistance.sentenceScore("this is nt corec")
		self.assertNotEqual(score, 1)

	def test_randomSentenceLowScore(self):
		ranText = ''.join(random.choice(string.printable) for i in range(45))
		scoreR = self.wordDistance.sentenceScore(ranText)
		self.assertLess(scoreR, 0.1, ranText)

	def test_reallyIncorrectLessThanNotSoIncorrect(self):
		score1 = self.wordDistance.sentenceScore("ths s nt corec")
		score2 = self.wordDistance.sentenceScore("ths is nt corec")
		self.assertGreater(score2, score1)

if __name__ == '__main__':
	unittest.main()
