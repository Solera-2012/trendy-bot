import unittest
import scraper
import json

class TestScraper(unittest.TestCase):
	def setUp(self):
		self.scrapy = scraper.Scraper()
		self.scrapy.authenticate()

	def test_get_tweets_from_user(self):
		tweets = list(self.scrapy.tweets_from_user("ropthe_", 3))
		self.assertEqual(len(tweets), 3)

	def test_get_tweets_from_hashtag(self):
		tweets = list(self.scrapy.tweets_from_hashtag("#math", 3))
		self.assertEqual(len(tweets), 3)

	def test_tests_from_hash_and_user_are_same_type(self):
		user_tweets = self.scrapy.tweets_from_user("ropthe_",2)
		hashtag_tweets = self.scrapy.tweets_from_hashtag("#math", 2)
		self.assertEqual(type(user_tweets), type(hashtag_tweets))

	def test_dump_tweets_to_file(self):
		tweets = self.scrapy.tweets_from_hashtag("#math", 2)
		self.scrapy.dump_tweets_to_file(tweets, "test.txt", "w+") 

		in_tweets = []
		for line in open("test.txt", "r"):
			in_tweets.append(json.loads(line))
		self.assertEqual(len(in_tweets), 2)


if __name__ == '__main__':
	unittest.main()
