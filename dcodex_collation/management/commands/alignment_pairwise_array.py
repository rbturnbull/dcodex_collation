import numpy as np
import pandas as pd

from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

class Command(BaseCommand):
    help = 'Gives a table showing the pairwise similarity by token for a set of manuscripts.'

    def add_arguments(self, parser):
        parser.add_argument('sigla', type=str,  nargs='+', help="The sigla of the manuscripts.")
        parser.add_argument('--start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('--end', type=str, help="The ending verse of the passage selection. If this is not given, then it only uses the start verse.")
        parser.add_argument('-k', '--skip', type=str, nargs='+', help="A list of verses to skip.")
        parser.add_argument('-f', '--file', type=str, help="An output CSV file.")
        parser.add_argument('-t', '--truncate', action='store_true', default=False, help="Truncate the output when printing to screen.")
        parser.add_argument('--atext', action='store_true', default=False, help="Includes the A-Text in the output.")

    def handle(self, *args, **options):
        
        manuscripts = [Manuscript.find(siglum) for siglum in options['sigla']]
        sigla = [ms.siglum for ms in manuscripts]

        
        VerseClass = manuscripts[0].verse_class()
        start_verse_string = options['start'] or ""
        end_verse_string = options['end'] or ""
        verses = VerseClass.queryset_from_strings( start_verse_string, end_verse_string )
        if options['skip']:
            verse_ids_to_skip = [VerseClass.get_from_string(verse_ref_to_skip).id for verse_ref_to_skip in options['skip']]
            verses = verses.exclude(id__in=verse_ids_to_skip)

        atext = options['atext']
        comparison_array = calc_pairwise_comparison_array(manuscripts, verses, atext=atext)        
        if atext:
            sigla += ["A-Text"]

        # Make a percentage
        comparison_array *= 100.0 
        df = pd.DataFrame(data=comparison_array, columns=sigla)
        df['MSS'] = sigla
        df = df.set_index('MSS')
        
        if options["file"]:
            df.to_csv(options["file"])

        if not options['truncate']:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

        print(df)
