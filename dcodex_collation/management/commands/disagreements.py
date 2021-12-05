from dcodex.models import *
from dcodex_collation.models import *

from django.core.management.base import BaseCommand

from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Finds unique readings for a group."

    def add_arguments(self, parser):
        parser.add_argument(
            "siglum1", type=str, help="The siglum for the first manuscript to compare."
        )
        parser.add_argument(
            "siglum2", type=str, help="The siglum for the second manuscript to compare."
        )
        self.add_verses_parser(parser, family_optional=True, start_optional=True)
        parser.add_argument("-o", "--output", type=str, help="An output CSV file.")

    def handle(self, *args, **options):
        siglum1 = options["siglum1"]
        siglum2 = options["siglum2"]
        manuscript1 = Manuscript.find(siglum1)
        if not manuscript1:
            raise Exception(f"Cannot find manuscript '{siglum1}'")
        manuscript2 = Manuscript.find(siglum2)
        if not manuscript2:
            if siglum2.lower().replace("-", "") == "atext":
                manuscript2 = None
            else:
                raise Exception(f"Cannot find manuscript '{siglum2}'")


        verse_class = manuscript1.verse_class()
        columns = self.get_columns_from_options(options, verse_class)

        disagreements_transitions_csv(
            manuscript1, manuscript2, columns=columns, dest=options["output"]
        )
