import os
from django.test import TestCase
import json

import scraper.services.TweetScraper

class TestTweetScraper(TestCase):
	def setUp(self):
		self.scrapy = TweetScraper.TweetScraper()
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

		os.remove("test.txt")

	def test_save_tweets_to_db(self):
		#grab tweets from hashtag
		math_tweets = self.scrapy.tweets_from_hashtag('#math', 1)

		for tweet in math_tweets:
		    #add tweet infomation to the database
		    self.scrapy.save_tweet_to_db(tweet, 'math')
