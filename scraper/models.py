from django.db import models

class Hashtag(models.Model):
    tag = models.CharField(max_length=50)

class Tweet(models.Model):
    tweet_text = models.CharField(max_length=200)
    time_created = models.DateTimeField('time tweeted')
    favorite_count = models.IntegerField(default=0)
    retweet_count = models.IntegerField(default=0)
    hastag = models.ForeignKey(Hashtag, on_delete=models.CASCADE)

    def __str__(self):
        return self.tweet_text

