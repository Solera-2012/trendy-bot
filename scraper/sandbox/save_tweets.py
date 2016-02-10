from secret_keys import secret_keys
import tweepy

#set up relative paths to access the database
import sys
import os.path
sys.path.append("../../")
sys.path.append('/homes/iws/mhsaul/venv/trendy-bot/trendy_site/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trendy_site.settings")
from scraper.models import Tweet

#load keys from secret file
key = secret_keys.secret_keys()

#authenticate
auth = tweepy.OAuthHandler(key.consumer_key, key.consumer_secret)
auth.set_access_token(key.access_token_key, key.access_token_secret)
api = tweepy.API(auth)

#grab tweets from user
new_tweets = api.user_timeline(screen_name="ropthe_", count=2)

#grab tweets from hashtag
math_tweets = tweepy.Cursor(api.search, q='#math').items(1)
for tweet in math_tweets:
    #add tweet infomation to the database
    t = Tweet(tweet_text=tweet.text, time_created=tweet.created_at.replace(tzinfo=None), \
				favorite_count=tweet.favorite_count, retweet_count=tweet.retweet_count)
    t.save()
