# Generated by Django 3.1.3 on 2021-06-13 01:23

from django.db import migrations
import django_extensions.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0008_auto_20210430_2318'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='transition',
            options={'ordering': ['column', 'transition_type']},
        ),
        migrations.AddField(
            model_name='transitiontype',
            name='slug',
            field=django_extensions.db.fields.AutoSlugField(blank=True, editable=False, populate_from=['name']),
        ),
    ]
