from django.core.management.base import BaseCommand, CommandError
from dcodex_collation.models import Alignment, State, Cell, Row
from dcodex.models import Manuscript
from dcodex_collation.graph import variant_graph
import networkx as nx
import matplotlib.pyplot as plt
import re


from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Separates different multi-word omissions into different states."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)
        # parser.add_argument("-o", "--output", type=str, help="An output file.")

    def handle(self, *args, **options):
        family, verses = self.get_family_and_verses_from_options(options)

        for verse in verses:
            alignment = Alignment.objects.filter(family=family, verse=verse).first()
            if not alignment:
                continue

            columns_count = alignment.column_set.count()
            G = variant_graph(alignment)

            # G = nx.convert_node_labels_to_integers(G, label_attribute="label")
            
            nodes = G.nodes(data=True)
            for start, end, data in G.edges(data=True):
                assert 'mss' in data
                # breakpoint()
                column_start = None
                column_end = None
                if nodes[start]['column_id'] == 0:
                    if nodes[end]['column_id'] == -1:
                        print("Whole verse omission")
                        column_start = 0
                        column_end = columns_count - 1
                    else:
                        assert nodes[end]['column']
                        if nodes[end]['column'].order != 0:
                            print("start omission")
                            column_start = 0
                            column_end = nodes[end]['column'].order
                elif nodes[end]['column_id'] == -1:
                    assert nodes[start]['column']
                    if nodes[start]['column'].order != columns_count -1:
                        print("end omission")
                        column_start = nodes[start]['column'].order
                        column_end = columns_count - 1
                else:
                    assert nodes[start]['column']
                    assert nodes[end]['column']
                    if nodes[start]['column'].order + 1 != nodes[end]['column'].order:
                        print("omission")
                        column_start = nodes[start]['column'].order
                        column_end = nodes[end]['column'].order

                if column_start is None:
                    continue

                # create state
                start_column = nodes[start]['column'].order if nodes[start]['column'] else "START"
                end_column = nodes[end]['column'].order if nodes[end]['column'] else "END"
                state_text = f"OMIT {start_column}→{end_column}"
                state, _ = State.objects.update_or_create(text=state_text)

                # Attach to mss at columns
                mss = Manuscript.objects.filter(siglum__in=data['mss'])
                rows = Row.objects.filter(transcription__manuscript__in=mss, alignment=alignment)
                cells = Cell.objects.filter(row__in=rows, column__order__gte=column_start, column__order__lte=column_end )
                cells.update(state=state)

                # print(start, end, data, nodes[start], nodes[end])
                

