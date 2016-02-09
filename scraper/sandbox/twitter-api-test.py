from secret_keys import secret_keys
import tweepy


#load keys from secret file
key = secret_keys.secret_keys()
print(key)

#authenticate
auth = tweepy.OAuthHandler(key.consumer_key, key.consumer_secret)
auth.set_access_token(key.access_token_key, key.access_token_secret)
api = tweepy.API(auth)

#grab tweets from user
new_tweets = api.user_timeline(screen_name="ropthe_", count=2)

#grab tweets from hashtag
math_tweets = tweepy.Cursor(api.search, q='#math').items(1)
for tweet in math_tweets:
	print(tweet)
