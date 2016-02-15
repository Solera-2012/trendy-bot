from django.test import TestCase

import scraper.services.SpellChecker 

class TestSpellChecker(TestCase):
	def setUp(self):
		self.sp = SpellChecker.SpellChecker()

	def test_wordsAreCorrected(self):
		word = self.sp.correct("feling")
		self.assertEqual(word, "feeling")

	def test_words(self): 
		text = "hi this is a word that I like to do"
		words = self.sp.words(text)
		words_list = ['hi', 'this', 'is', 'a', 'word', 'that', 'i', 'like', 'to', 'do']
		self.assertEqual(words_list, words)		

	def test_train(self):
		pass

	def edits1(self):
		pass

	def known_edits2(self):
		pass
	
	def known(self): 
		pass

	def correct(self):
		pass
