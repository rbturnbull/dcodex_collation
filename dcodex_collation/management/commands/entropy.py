from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from django.db.models import F
from django.db.models import Count, Sum
from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Calculates the entropy for each column."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)
        parser.add_argument(
            "-i",
            "--ignore",
            action="store_true",
            default=False,
            help="Whether it should ignore transisions in the TransitionsToIgnore group.",
        )
        parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            default=False,
            help="Whether it should only output results for all columns instead of just columns with multiple states.",
        )
        parser.add_argument(
            "--base", 
            type=float, 
            help="The base to use for the logarithm.",
            default=None,
        )

    def handle(self, *args, **options):
        allow_ignore = options["ignore"]
        base = options["base"]
        all = options["all"]

        columns = self.get_columns_from_options(options).filter(atext=None)
        print("index", "verse", "column", "entropy", sep=",")
        for index, column in enumerate(columns):
            entropy = column.entropy(base=base, allow_ignore=allow_ignore)
            if all or entropy > 0.0:
                print(index, column.alignment.verse, column.order, entropy, sep=",")


