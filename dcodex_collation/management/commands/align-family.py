from django.core.management.base import BaseCommand, CommandError
from pathlib import Path
from dcodex.models import Family
from dcodex_collation.models import Alignment, align_family_at_verse, update_alignment
from distutils.util import strtobool

from ._mixins import VersesCommandMixin

class Command(VersesCommandMixin, BaseCommand):
    help = 'Aligns all witnesses in a range of verses.'

    def add_arguments(self, parser):
        self.add_verses_parser(parser, start_optional=True)
        parser.add_argument('--replace', type=strtobool, nargs='?', const=True, default=False, help="If true, this replaces any alignment if it exists. (Default false)")

    def handle(self, *args, **options):
        family, verses = self.get_family_and_verses_from_options(options)    

        gotoh_param = [6.6995597099885345, -0.9209875054657459, -5.097397327423096, -1.3005714416503906]        
        for verse in verses:
            print("Aligning:", verse)
            if options['replace']:
                Alignment.objects.filter( verse=verse, family=family ).delete()

            alignment = Alignment.objects.filter( verse=verse, family=family ).first()
            if not alignment:
                alignment = align_family_at_verse( family, verse, gotoh_param )    
            else:
                update_alignment(alignment, gotoh_param=gotoh_param)