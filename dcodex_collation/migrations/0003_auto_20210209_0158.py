# Generated by Django 3.1.3 on 2021-02-09 09:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0002_auto_20201215_0309'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='row',
            options={'ordering': ['transcription__manuscript']},
        ),
    ]
