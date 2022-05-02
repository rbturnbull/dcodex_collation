import csv
import sys
from django.core.management.base import BaseCommand, CommandError
from pathlib import Path
from dcodex.models import Family
from dcodex_collation.models import State, Cell, Column, align_family_at_verse, update_alignment
from distutils.util import strtobool
from django.contrib.sites.models import Site

from ._mixins import VersesCommandMixin

class Command(VersesCommandMixin, BaseCommand):
    help = 'Searches for text in the cells of an alignment.'

    def add_arguments(self, parser):
        self.add_verses_parser(parser, start_optional=True)
        parser.add_argument('regex', type=str, help="The regex text to search for.")
        parser.add_argument("-o", "--output", type=str, help="An output CSV file.")

    def handle(self, *args, **options):
        alignments = self.get_alignments_from_options(options)
        site = Site.objects.get_current()
        domain_name = site.domain

        states = State.objects.filter(text__regex=options['regex'])
        cells = Cell.objects.filter(state__in=states, column__alignment__in=alignments)
        columns = Column.objects.filter(cell__in=cells).distinct()

        delimiter = "\t"
        csv_writers = [csv.writer(sys.stdout, delimiter=delimiter)]
        file = None
        if options["output"]:
            file = open(options["output"], "w", newline="")
            csv_writers.append(csv.writer(file, delimiter=delimiter))

        for csv_writer in csv_writers:
            csv_writer.writerow(
                [
                    "verse:column",
                    "url",
                    "states",
                    "mss",
                    "other_states",
                    "other_mss",
                ]
            )            
            for column in columns:
                column_cells = cells.filter(column=column)
                state_texts = set(column_cells.values_list("state__text", flat=True))
                mss = column_cells.values_list("row__transcription__manuscript__siglum", flat=True)
                all_states = State.objects.filter(cell__column=column)
                other_states = all_states.difference(states)
                other_states_texts = set([text or "OMIT" for text in other_states.values_list("text", flat=True)])
                other_states_cells = Cell.objects.filter(column=column, state__in=list(other_states))
                other_states_mss = other_states_cells.values_list("row__transcription__manuscript__siglum", flat=True)
                csv_writer.writerow(
                    [
                        f"{column.alignment.verse}:{column.order}",
                        f"http://{domain_name}{column.get_absolute_url()}",
                        "|".join(state_texts),
                        " ".join(mss),
                        "|".join(other_states_texts),
                        " ".join(other_states_mss),
                    ]
                )

        if file:
            file.close()            