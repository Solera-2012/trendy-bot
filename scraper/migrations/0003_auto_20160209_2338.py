# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0002_hashtag'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='hashtag',
            name='tweets',
        ),
        migrations.AddField(
            model_name='tweet',
            name='hastag',
            field=models.ForeignKey(to='scraper.Hashtag', default=None),
            preserve_default=False,
        ),
    ]
