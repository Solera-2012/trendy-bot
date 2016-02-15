import WordDistance 
import unittest

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


if __name__ == '__main__':
	unittest.main()
