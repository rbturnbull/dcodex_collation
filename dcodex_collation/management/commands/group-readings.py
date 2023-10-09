import sys
import csv
from django.contrib.sites.models import Site
from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from ._mixins import VersesCommandMixin


def get_equivalent_state_ids(states, transitions_to_process):
    """Returns the set of state ids with the states with the lowest ids if transitions between states can be ignored."""
    equivalent_state_ids = set()
    for state in states:
        while True:
            if transition := transitions_to_process.filter(
                start_state=state, start_state__id__gt=F("end_state__id")
            ).first():
                state = transition.end_state
            elif transition := transitions_to_process.filter(
                end_state=state, end_state__id__gt=F("start_state__id")
            ).first():
                state = transition.start_state
            else:
                break
        equivalent_state_ids.update([state.id])
    return equivalent_state_ids


class Command(VersesCommandMixin, BaseCommand):
    help = "Finds readings for a group."

    def add_arguments(self, parser):
        parser.add_argument(
            "sigla", type=str, nargs="+", help="The sigla for the mss in the group."
        )
        self.add_verses_parser(parser, family_optional=True, start_optional=True)
        parser.add_argument("-o", "--output", type=str, help="An output CSV file.")
        parser.add_argument(
            "-i",
            "--ignore",
            action="store_true",
            default=False,
            help="Whether it should ignore transisions in the TransitionsToIgnore group.",
        )
        parser.add_argument(
            "exclude", type=str, nargs="+", help="The sigla for the mss to exclude from the outgroup."
        )


    def handle(self, *args, **options):

        allow_ignore = options["ignore"]

        sigla = options["sigla"]
        mss_in_group = [Manuscript.find(siglum) for siglum in sigla]
        if not mss_in_group:
            print("No manuscripts found.")

        mss_ids_in_group = [ms.id for ms in mss_in_group]
        
        # Make mss_in_group a queryset so that we can use it to filter
        mss_in_group = Manuscript.objects.filter(id__in=mss_ids_in_group)

        outgroup_exclude = set(mss_ids_in_group)
        if options["exclude"]:
            outgroup_exclude.update([Manuscript.find(siglum).id for siglum in options["exclude"]])

        verse_class = mss_in_group[0].verse_class()

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
                    "state",
                    "group_count",
                    "group_agree",
                    "group_disagree",
                    "group_disagree_states",
                    "group_missing",
                    "outgroup_count",
                    "outgroup_agree",
                    "outgroup_disagree",
                    "outgroup_disagree_states",
                    "difference",
                ]
            )

        site = Site.objects.get_current()
        domain_name = site.domain

        if allow_ignore:
            transition_type_ids_to_ignore = (
                TransitionTypeToIgnore.objects.all().values_list(
                    "transition_type__id", flat=True
                )
            )

        alignments = self.get_alignments_from_options(options, verse_class=verse_class)
        for alignment in alignments:
            for column in alignment.column_set.all():
                cells_ingroup = column.cell_set.filter(
                    row__transcription__manuscript__id__in=mss_ids_in_group
                )
                cells_outgroup = column.cell_set.exclude(
                    row__transcription__manuscript__id__in=outgroup_exclude
                )

                states_ingroup = State.objects.filter(cell__in=cells_ingroup).distinct()
                if states_ingroup.count() == 0:
                    continue
                states_outgroup = State.objects.filter(
                    cell__in=cells_outgroup
                ).distinct()
                if states_outgroup.count() == 0:
                    continue

                states_ingroup_list = list(states_ingroup)
                states_ingroup_counts = np.array(
                    [
                        cells_ingroup.filter(state=state).count()
                        for state in states_ingroup_list
                    ]
                )
                max_count = np.max(states_ingroup_counts)

                if np.count_nonzero(states_ingroup_counts == max_count) > 1:
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
                group_agree = [
                    cell.row.transcription.manuscript.siglum
                    for cell in cells_ingroup.filter(state=state)
                ]
                group_disagree = [
                    cell.row.transcription.manuscript.siglum
                    for cell in cells_ingroup.exclude(state=state)
                ]

                outgroup_agree = [
                    cell.row.transcription.manuscript.siglum
                    for cell in cells_outgroup.filter(state=state)
                ]
                outgroup_disagree = [
                    cell.row.transcription.manuscript.siglum
                    for cell in cells_outgroup.exclude(state=state)
                ]
                outgroup_count = len(outgroup_agree)

                extant_mss_ids = alignment.row_set.filter(
                    transcription__manuscript__id__in=mss_ids_in_group
                ).values_list("transcription__manuscript__id")
                missing_mss = mss_in_group.exclude(id__in=extant_mss_ids)
                missing_mss_sigla = [ms.siglum for ms in missing_mss]

                group_disagree_states = states_ingroup.exclude(id=state.id)
                outgroup_disagree_states = states_outgroup.exclude(id=state.id)

                for csv_writer in csv_writers:
                    csv_writer.writerow(
                        [
                            f"{column.alignment.verse}:{column.order}",
                            f"http://{domain_name}{column.alignment.get_absolute_url()}",
                            state,
                            group_count,
                            "/".join(group_agree),
                            "/".join(group_disagree),
                            "/".join([str(state) for state in group_disagree_states]),
                            "/".join(missing_mss_sigla),
                            outgroup_count,
                            "/".join(outgroup_agree),
                            "/".join(outgroup_disagree),
                            "/".join(
                                [str(state) for state in outgroup_disagree_states]
                            ),
                            group_count - outgroup_count,
                        ]
                    )

        if file:
            file.close()
