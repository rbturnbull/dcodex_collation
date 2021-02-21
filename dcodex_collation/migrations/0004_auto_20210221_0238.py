# Generated by Django 3.1.3 on 2021-02-21 10:38

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('dcodex_collation', '0003_auto_20210209_0158'),
    ]

    operations = [
        migrations.CreateModel(
            name='TransitionClassifier',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('polymorphic_ctype', models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_dcodex_collation.transitionclassifier_set+', to='contenttypes.contenttype')),
                ('transition_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dcodex_collation.transitiontype')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
        ),
        migrations.CreateModel(
            name='RegexTransitionClassifier',
            fields=[
                ('transitionclassifier_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='dcodex_collation.transitionclassifier')),
                ('start_state_regex', models.CharField(max_length=255)),
                ('end_state_regex', models.CharField(max_length=255)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('dcodex_collation.transitionclassifier',),
        ),
        migrations.AddField(
            model_name='transition',
            name='classifier',
            field=models.ForeignKey(blank=True, default=None, help_text='The transition classifer used to automatically assign the transition type for these states.', null=True, on_delete=django.db.models.deletion.SET_DEFAULT, to='dcodex_collation.transitionclassifier'),
        ),
    ]
