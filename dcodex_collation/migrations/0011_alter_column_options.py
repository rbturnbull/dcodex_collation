# Generated by Django 3.2.9 on 2021-12-05 11:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0010_auto_20210612_1827'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='column',
            options={'ordering': ['alignment__verse__rank', 'order']},
        ),
    ]