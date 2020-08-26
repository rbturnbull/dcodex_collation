# Generated by Django 3.0.8 on 2020-08-26 12:02

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dcodex_collation', '0014_auto_20200826_2202'),
    ]

    operations = [
        migrations.CreateModel(
            name='Transition',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('inverse', models.BooleanField()),
                ('column', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dcodex_collation.Column')),
                ('end_state', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='end_state', to='dcodex_collation.State')),
                ('start_state', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='start_state', to='dcodex_collation.State')),
                ('transition_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dcodex_collation.TransitionType')),
            ],
        ),
    ]
