from dcodex.models import * 
from dcodex_collation.models import *

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Finds unique readings for a group."

    def add_arguments(self, parser):
        parser.add_argument('siglum1', type=str, help="The siglum for the first manuscript to compare.")
        parser.add_argument('siglum2', type=str, help="The siglum for the second manuscript to compare.")
        parser.add_argument('-s', '--start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('-e', '--end', type=str, help="The ending verse of the passage selection.")
        parser.add_argument('-f', '--file', type=str, help="An output file.")

    def handle(self, *args, **options):
        siglum1 = options['siglum1']
        siglum2 = options['siglum2']
        manuscript1 = Manuscript.find(siglum1)
        if not manuscript1:
            raise Exception(f"Cannot find manuscript '{siglum1}'")
        manuscript2 = Manuscript.find(siglum2)
        if not manuscript2:
            if siglum2.lower().replace("-","") == "atext":
                manuscript2 = None
            else:
                raise Exception(f"Cannot find manuscript '{siglum2}'")

        start_verse_string = options['start'] or ""
        end_verse_string = options['end'] or ""

        VerseClass = manuscript1.verse_class()
        if manuscript2:
            assert manuscript1.verse_class() == manuscript2.verse_class()
        verses = VerseClass.queryset_from_strings( start_verse_string, end_verse_string )

        disagreements_transitions_csv(manuscript1, manuscript2, verses=verses, dest=options['file'])
