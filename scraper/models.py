from django.db import models

# Create your models here.
class Tweet(models.Model):
    tweet_text = models.CharField(max_length=200)
    time_created = models.DateTimeField('time tweeted')
    favorite_count = models.IntegerField(default=0)
    retweet_count = models.IntegerField(default=0)

    def __str__(self):
        return self.tweet_text
