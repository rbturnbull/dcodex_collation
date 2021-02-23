from django.core.management.base import BaseCommand, CommandError
import pandas as pd
from dcodex.models import *
from dcodex_collation.models import *


class Command(BaseCommand):
    help = 'Creates TransitionClassifier objects from a spreadsheet.'

    def add_arguments(self, parser):
        parser.add_argument('CSV', type=str, help="A CSV file with the start state regex, end state regex and the transition type.")

    def handle(self, *args, **options):
        df = pd.read_csv( options['CSV'] )
        print(df)

        for index, row in df.iterrows():
            # print(row)
            transition_type = TransitionType.objects.filter(name=row['transition_type']).first()
            start_state_regex, end_state_regex = row['start_state_regex'], row['end_state_regex']
            if transition_type == None:
                transition_type = TransitionType.objects.filter(inverse_name=row['transition_type']).first()
                start_state_regex, end_state_regex = end_state_regex, start_state_regex

            if transition_type == None:
                raise Exception("Cannot find transition type {transition_type}")
                
                            
            if 'name' in row:
                classifier_name = row['name']
            else:
                classifier_name = f"{start_state_regex} -> {end_state_regex}: {transition_type.name}"

            print(classifier_name, transition_type, start_state_regex, end_state_regex)
            RegexTransitionClassifier.objects.update_or_create(  
                start_state_regex=start_state_regex,
                end_state_regex=end_state_regex,
                name=classifier_name,
                transition_type=transition_type,
            )