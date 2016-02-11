from secret_keys import secret_keys
import tweepy
import tweet_scraper

#set up relative paths to access the database
'''import sys
import os
sys.path.append(os.path.realpath('../..'))
sys.path.append(os.path.realpath('../..') + '/trendy_site/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trendy_site.settings")
from scraper.models import Tweet, Hashtag
import django
django.setup()'''

s = tweet_scraper.Scraper()

#authenticate
s.authenticate()

#grab tweets from hashtag
math_tweets = s.tweets_from_hashtag('#math', 1)

for tweet in math_tweets:
    #add tweet infomation to the database
    s.save_tweet_to_db(tweet, 'math')
