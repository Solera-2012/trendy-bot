# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('scraper', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Hashtag',
            fields=[
                ('id', models.AutoField(auto_created=True, serialize=False, verbose_name='ID', primary_key=True)),
                ('tag', models.CharField(max_length=50)),
                ('tweets', models.ForeignKey(to='scraper.Tweet')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
