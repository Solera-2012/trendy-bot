from secret_keys import secret_keys
import tweepy

key = secret_keys.secret_keys()

auth = tweepy.OAuthHandler(key.consumer_key, key.consumer_secret)
auth.set_access_token(key.access_token_key, key.access_token_secret)

api = tweepy.API(auth)
public_tweets = api.home_timeline()
for tweet in public_tweets:
	print(tweet.text)


