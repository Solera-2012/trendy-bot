import tweepy
import json
from secret_keys import secret_keys

class Scraper():
	#load keys from secret file
	key = secret_keys.secret_keys()

	def __init__(self):
		pass

	def authenticate(self):
		auth = tweepy.OAuthHandler(self.key.consumer_key, self.key.consumer_secret)
		auth.set_access_token(self.key.access_token_key, self.key.access_token_secret)
		self.api = tweepy.API(auth)

	def tweets_from_user(self, user, count):
		return tweepy.Cursor(self.api.user_timeline, screen_name=user).items(count)

	def tweets_from_hashtag(self, hashtag, count):
		return tweepy.Cursor(self.api.search, q=hashtag).items(count)

	def save_tweet_to_db(self, tweet):
		t = Tweet(tweet_text=tweet.text, \
				time_created = tweet.created_at.replace(tzinfo=None), \
				favorite_count=tweet.favorite_count, \
				retweet_count=tweet.retweet_count)
		t.save()

	def dump_tweets_to_file(self, tweets, filename, to="w"):
		assert(to[0] == "w")
		with open(filename, to) as outfile:
			for tweet in tweets:
				outfile.write("["+json.dumps(tweet._json)+"]\n")

	




