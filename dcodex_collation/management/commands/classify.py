from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *


class Command(BaseCommand):
    help = 'Classifies transitions with a list of TransitionClassifier objects.'

    def add_arguments(self, parser):
        parser.add_argument('family', type=str, help="The siglum for a family of manuscripts.")
        parser.add_argument('classifiers', type=str, nargs="?", help="The classifiers to use. If none specified, then all are used from the database.")
        parser.add_argument('--start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('--end', type=str, help="The ending verse of the passage selection. If this is not given, then it only uses the start verse.")

    def handle(self, *args, **options):
        family = Family.objects.get(name=options['family'])
        witnesses_in_family = family.manuscripts()

        VerseClass = witnesses_in_family.first().verse_class()

        start_verse_string = options['start'] or ""
        end_verse_string = options['end'] or ""

        classifiers = TransitionClassifier.objects.all()
        if options['classifiers']:
            classifiers = classifiers.filter(name__in=[options['classifiers']])

        if classifiers.count() == 0:
            print("No classifiers found.")
            # return

        verses = VerseClass.queryset_from_strings( start_verse_string, end_verse_string )

        alignments = Alignment.objects.filter(family=family, verse__in=verses)
        for alignment in alignments:
            for column in alignment.column_set.all():
                for pair_rank, pair in enumerate(column.state_pairs()):
                        
                    if not column.transition_for_pair(pair_rank):
                        start_state = pair[0]
                        end_state = pair[1]
                        print(column, pair_rank, start_state, end_state)

                        for classifier in classifiers:
                            transition = classifier.classify( column, start_state, end_state )
                            
                            if transition:
                                print(f"classifier {classifier} matched.")
                                break


