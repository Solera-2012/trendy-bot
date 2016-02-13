import tweet_scraper

s = tweet_scraper.Scraper()

#authenticate
s.authenticate()

#grab tweets from hashtag
math_tweets = s.tweets_from_hashtag('#math', 1)

for tweet in math_tweets:
    #add tweet infomation to the database
    s.save_tweet_to_db(tweet, 'math')
