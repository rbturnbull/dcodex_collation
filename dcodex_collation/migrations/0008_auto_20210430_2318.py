# Generated by Django 3.1.3 on 2021-05-01 06:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("dcodex_collation", "0007_auto_20210407_0437"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="column",
            options={"ordering": ["alignment", "order"]},
        ),
    ]
