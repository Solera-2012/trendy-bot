import pytz
import tweepy
import json
import sys
import os
sys.path.append(os.path.realpath('../..'))
sys.path.append(os.path.realpath('../..') + '/trendy_site/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trendy_site.settings")
import django
from secret_keys import secret_keys
from scraper.models import Tweet, Hashtag


#what is this for?
django.setup()

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

    def save_tweet_to_db(self, tweet, hashtag):
        #the tz is our time, not the time that the tweet was created. what does time_created track?
		tz = pytz.timezone('America/Los_Angeles')

        if not Hashtag.objects.filter(tag=hashtag).exists():
            h = Hashtag(tag=hashtag)
            h.save()
        else:
            h = Hashtag.objects.get(tag=hashtag)

        if not h.tweet_set.filter(tweet_text=tweet.text).exists():
            h.tweet_set.create(tweet_text=tweet.text, \
                    time_created=tz.localize(tweet.created_at), \
                    favorite_count=tweet.favorite_count, \
                    retweet_count=tweet.retweet_count)
		else:
			#add information to indicate that the tweet is already seen - 
			#this might happen if we get retweets - we want to keep information that it has been processed
			#additionally - we need to know the difference between a tweet we have seen and a retweet / identical tweet


    def dump_tweets_to_file(self, tweets, filename, to="w"):
        assert(to[0] == "w")#intend we only write or overwrite (w or w+)
        with open(filename, to) as outfile:
            for tweet in tweets:
        	    outfile.write("["+json.dumps(tweet._json)+"]\n")

	




