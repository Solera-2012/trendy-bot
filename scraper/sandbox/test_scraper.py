import unittest
import scraper
import json

class TestScraper(unittest.TestCase):
	def setUp(self):
		self.scrapy = scraper.Scraper()
		self.scrapy.authenticate()
	
	def test_get_tweets_from_user(self):
		tweets = self.scrapy.tweets_from_user("ropthe_", 3)
		self.assertEqual(len(tweets), 3)

	def test_get_tweets_from_hashtag(self):
		tweets = list(self.scrapy.tweets_from_hashtag("#math", 3))
		self.assertEqual(len(tweets), 3)


	def test_dump_tweets_to_file(self):
		tweets = self.scrapy.tweets_from_hashtag("#math", 2)
		
		self.scrapy.dump_tweets_to_file(tweets, "test.txt") 



if __name__ == '__main__':
	unittest.main()
