# Generated by Django 3.0.8 on 2020-08-27 03:08

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0015_transition'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='state',
            name='column',
        ),
    ]