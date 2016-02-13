import sys
import os
# add scraper to the system path
sys.path.append(os.path.realpath('../../..'))
sys.path.append(os.path.realpath('../../..') + '/trendy_site/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trendy_site.settings")
from scraper.models import Tweet
# allows access to site database
import django
django.setup()

tweets = Tweet.objects.all()
with open('corpus.txt', 'w') as f:
    for t in tweets:
        f.write(t.tweet_text + '\n')

