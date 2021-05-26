import sys
import csv
from django.contrib.sites.models import Site

from dcodex.models import * 
from dcodex_collation.models import *


from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = "Finds unique readings for a group."

    def add_arguments(self, parser):
        parser.add_argument('family', type=str, help="The siglum for a family of manuscripts.")
        parser.add_argument('sigla', type=str, nargs="+", help="The sigla for the mss in the group.")
        parser.add_argument('--start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('--end', type=str, help="The ending verse of the passage selection. If this is not given, then it only aligns the start verse.")
        parser.add_argument('--file', type=str, help="An output file.")

    def handle(self, *args, **options):
        family = Family.objects.get(name=options['family'])
        witnesses_in_family = family.manuscripts()

        allow_ignore = True

        sigla = options['sigla']
        mss_in_group = [Manuscript.find(siglum) for siglum in sigla]
        mss_ids_in_group = [ms.id for ms in mss_in_group]

        VerseClass = witnesses_in_family.first().verse_class()

        start_verse_string = options['start'] or ""
        end_verse_string = options['end'] or ""

        verses = VerseClass.queryset_from_strings( start_verse_string, end_verse_string )

        delimiter = "\t"
        csv_writers = [csv.writer(sys.stdout, delimiter=delimiter)]
        file = None
        if options["file"]:
            file = open(options["file"], "w", newline='')
            csv_writers.append(csv.writer(file, delimiter=delimiter))

        for csv_writer in csv_writers:
            csv_writer.writerow([
                "verse",
                "column",
                "states_in_group",
                "states_out_of_group",
                "url",
            ])

        site = Site.objects.get_current()
        domain_name = site.domain

        alignments = Alignment.objects.filter(family=family, verse__in=verses)
        for alignment in alignments:
            for column in alignment.column_set.all():
                cells_ingroup = column.cell_set.filter(row__transcription__manuscript__id__in=mss_ids_in_group)
                cells_outgroup = column.cell_set.exclude(row__transcription__manuscript__id__in=mss_ids_in_group)
                
                states_ingroup = State.objects.filter(cell__in=cells_ingroup).distinct()
                if states_ingroup.count() == 0:
                    continue
                states_outgroup = State.objects.filter(cell__in=cells_outgroup).distinct()
                if states_outgroup.count() == 0:
                    continue

                intersection = states_ingroup & states_outgroup
                if intersection.count() > 0:
                    continue

                for csv_writer in csv_writers:
                    csv_writer.writerow([
                        str(column.alignment.verse), 
                        column.order, 
                        "/".join([str(state) for state in states_ingroup]),
                        "/".join([str(state) for state in states_outgroup]),
                        f"http://{domain_name}{column.alignment.get_absolute_url()}",
                    ])

        if file:
            file.close()


