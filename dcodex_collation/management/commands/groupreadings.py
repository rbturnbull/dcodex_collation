import sys
import csv
from django.contrib.sites.models import Site

from dcodex.models import * 
from dcodex_collation.models import *


from django.core.management.base import BaseCommand, CommandError

def get_equivalent_state_ids( states, transitions_to_process ):
    """ Returns the set of state ids with the states with the lowest ids if transitions between states can be ignored. """
    equivalent_state_ids = set()
    for state in states:
        while True:
            if transition := transitions_to_process.filter(start_state=state, start_state__id__gt=F("end_state__id")).first():
                state = transition.end_state
            elif transition := transitions_to_process.filter(end_state=state, end_state__id__gt=F("start_state__id")).first():
                state = transition.start_state
            else:
                break
        equivalent_state_ids.update([state.id])
    return equivalent_state_ids


class Command(BaseCommand):
    help = "Finds readings for a group."

    def add_arguments(self, parser):
        parser.add_argument('family', type=str, help="The siglum for a family of manuscripts.")
        parser.add_argument('sigla', type=str, nargs="+", help="The sigla for the mss in the group.")
        parser.add_argument('-s', '--start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('-e', '--end', type=str, help="The ending verse of the passage selection. If this is not given, then it only aligns the start verse.")
        parser.add_argument('-f', '--file', type=str, help="An output file.")
        parser.add_argument('-i', '--ignore', action='store_true', default=False, help="Whether it should ignore transisions in the TransitionsToIgnore group.")

    def handle(self, *args, **options):
        family = Family.objects.get(name=options['family'])
        witnesses_in_family = family.manuscripts()

        allow_ignore = options['ignore']

        sigla = options['sigla']
        mss_in_group = [Manuscript.find(siglum) for siglum in sigla]
        mss_ids_in_group = [ms.id for ms in mss_in_group]
        mss_in_group = Manuscript.objects.filter(id__in=mss_ids_in_group)

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
                "state",
                "group_count",
                "group_agree",
                "group_disagree",
                "outgroup_count",
                "outgroup_agree",
                "outgroup_disagree",
                "difference",
                "url",
            ])

        site = Site.objects.get_current()
        domain_name = site.domain
        
        if allow_ignore:
            transition_type_ids_to_ignore = TransitionTypeToIgnore.objects.all().values_list("transition_type__id", flat=True)

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
                
                states_ingroup_list = list(states_ingroup)
                states_ingroup_counts = np.array([cells_ingroup.filter(state=state).count() for state in states_ingroup_list])
                max_count = np.max(states_ingroup_counts)

                if np.count_nonzero( states_ingroup_counts == max_count ) > 1:
                    continue
                
                state_index = np.argmax(states_ingroup_counts)
                state = states_ingroup_list[state_index]

                # if allow_ignore:
                #     transitions_to_process = Transition.objects.filter(transition_type__id__in=transition_type_ids_to_ignore, column=column)
                    
                #     equivalent_state_ids_ingroup = get_equivalent_state_ids( states_ingroup, transitions_to_process)
                #     equivalent_state_ids_outgroup = get_equivalent_state_ids( states_outgroup, transitions_to_process)

                # extant_mss_ids = alignment.row_set.filter(transcription__manuscript__id__in=mss_ids_in_group).values_list('transcription__manuscript__id')
                # missing_mss = mss_in_group.exclude(id__in=extant_mss_ids)
                # missing_mss_sigla = [ms.siglum for ms in missing_mss]

                group_count = max_count
                group_agree = [cell.row.transcription.manuscript.siglum for cell in cells_ingroup.filter(state=state)]
                group_disagree = [cell.row.transcription.manuscript.siglum for cell in cells_ingroup.exclude(state=state)]

                outgroup_agree = [cell.row.transcription.manuscript.siglum for cell in cells_outgroup.filter(state=state)]
                outgroup_disagree = [cell.row.transcription.manuscript.siglum for cell in cells_outgroup.exclude(state=state)]
                outgroup_count = len(outgroup_agree)
                
                for csv_writer in csv_writers:
                    csv_writer.writerow([
                        str(column.alignment.verse), 
                        column.order, 
                        state,
                        group_count,
                        "/".join(group_agree),
                        "/".join(group_disagree),
                        outgroup_count,
                        "/".join(outgroup_agree),
                        "/".join(outgroup_disagree),
                        group_count-outgroup_count,
                        f"http://{domain_name}{column.alignment.get_absolute_url()}",
                    ])

        if file:
            file.close()


