# Generated by Django 3.0.8 on 2020-08-26 00:14

from django.db import migrations, models
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0012_auto_20200826_0907'),
    ]

    operations = [
        migrations.AlterField(
            model_name='alignment',
            name='id_to_word',
            field=models.BinaryField(blank=True, help_text='Index of vocab dictionary', null=True),
        ),
        migrations.AlterField(
            model_name='alignment',
            name='word_to_id',
            field=jsonfield.fields.JSONField(blank=True, help_text='Vocab dictionary', null=True),
        ),
        migrations.AlterField(
            model_name='row',
            name='tokens',
            field=models.BinaryField(blank=True, help_text='Numpy array for the tokens. IDs correspond to the vocab in the alignment', null=True),
        ),
    ]