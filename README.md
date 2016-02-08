# Trendy-bot

Trendy-bot is made of 3 parts:
* Scraper that will find trending hashtags and collect tweets in these groups
* Creator that will generate new tweets based on the collected tweets
* Tracker that will gather statistics describing how the public is responding to the tweets.


## Scraper
* Find trending hashtags
* gather tweets with these hashtags
* process these tweets
  * clean the tweets, remove @s and #s that are no needed.
  * gather stats on these tweets; retweets, followers of user to tweeted it, part of a conversation?
* store the tweets in a catalog

## Creator
* Train multiple models on the tweet catalog
* Generate new tweets using these models

## Tracker
* Choose the best tweets of the creator's tweets
* Tweet those tweets
* Track statics about response to these tweets
* Base item one's choice on these statistics




