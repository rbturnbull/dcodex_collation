# Generated by Django 3.1.3 on 2020-12-15 11:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dcodex_collation", "0001_squashed_0021_remove_row_tokens"),
    ]

    operations = [
        migrations.AlterField(
            model_name="token",
            name="rank",
            field=models.PositiveIntegerField(),
        ),
    ]
