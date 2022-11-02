from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from django.db.models import F
from django.db.models import Count, Sum
from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Sets the A-Text automatically if there are only singular readings as alternatives."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)
        parser.add_argument(
            "-i",
            "--ignore",
            action="store_true",
            default=False,
            help="Whether it should ignore transisions in the TransitionsToIgnore group.",
        )

    def handle(self, *args, **options):
        allow_ignore = options["ignore"]

        columns = self.get_columns_from_options(options).filter(atext=None)
        for column in columns:
            states = column.states(allow_ignore=allow_ignore)
            if len(states) != 2:
                continue

            counts = [len(state.cells_at(column)) for state in states]

            if counts[0] == 1 and counts[1] > 1:
                column.atext = states[1]
                other = states[0]
            elif counts[1] == 1 and counts[0] > 1:
                column.atext = states[0]
                other = states[1]
            else:
                continue
                
            column.atext_notes = f"Setting A-Text automatically because '{other.str_at(column)}' is a singular reading."
            print(column, "A-Text:", column.atext, "-", column.atext_notes)


