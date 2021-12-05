import numpy as np
import pandas as pd

from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from ._mixins import VersesCommandMixin

class Command(VersesCommandMixin, BaseCommand):
    help = "Gives a table showing the pairwise similarity by token for a set of manuscripts."

    def add_arguments(self, parser):
        parser.add_argument(
            "sigla", type=str, nargs="+", help="The sigla of the manuscripts."
        )
        self.add_verses_parser(parser, family_optional=True, start_optional=True)
        parser.add_argument("-o", "--output", type=str, help="An output CSV file.")
        parser.add_argument(
            "-t",
            "--truncate",
            action="store_true",
            default=False,
            help="Truncate the output when printing to screen.",
        )
        parser.add_argument(
            "--atext",
            action="store_true",
            default=False,
            help="Includes the A-Text in the output.",
        )

    def handle(self, *args, **options):
        manuscripts = [Manuscript.find(siglum) for siglum in options["sigla"]]
        sigla = [ms.siglum for ms in manuscripts]

        verse_class = manuscripts[0].verse_class()
        columns = self.get_columns_from_options(options, verse_class)

        atext = options["atext"]
        comparison_array = calc_pairwise_comparison_array(
            manuscripts, columns=columns, atext=atext
        )
        if atext:
            sigla += ["A-Text"]

        # Make a percentage
        comparison_array *= 100.0
        df = pd.DataFrame(data=comparison_array, columns=sigla)
        df["MSS"] = sigla
        df = df.set_index("MSS")

        if options["output"]:
            df.to_csv(options["output"])

        if not options["truncate"]:
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", None)

        print(df)
