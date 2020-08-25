# Generated by Django 3.0.8 on 2020-08-25 07:17

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0004_atext_transition_transitiontype'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='transitiontype',
            options={'ordering': ['name']},
        ),
        migrations.AddField(
            model_name='transition',
            name='inverse',
            field=models.BooleanField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='transition',
            name='transition_type',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='dcodex_collation.TransitionType'),
            preserve_default=False,
        ),
    ]
