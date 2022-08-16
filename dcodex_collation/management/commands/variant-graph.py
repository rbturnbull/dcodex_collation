from django.core.management.base import BaseCommand, CommandError
from dcodex_collation.models import Alignment
from dcodex_collation.graph import variant_graph
import networkx as nx
import matplotlib.pyplot as plt
import re


from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Creates a variant graph from an alignment."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)
        # parser.add_argument("-o", "--output", type=str, help="An output file.")

    def handle(self, *args, **options):
        family, verses = self.get_family_and_verses_from_options(options)

        for verse in verses:
            alignment = Alignment.objects.filter(family=family, verse=verse).first()
            if not alignment:
                continue

            G = variant_graph(alignment)

            G = nx.convert_node_labels_to_integers(G, label_attribute="label")
            # nx.draw_circular(G, with_labels = True)
            # nx.draw_spectral(G, with_labels = True)
            for node, data in G.nodes(data=True):
                data['label'] = re.sub(r"\d+\-(\D+)", r"\1", data['label'].replace("-None", "-OMIT") ).replace("+", " ")
                # data.clear()
            
            # nx.draw_spring(G, with_labels = True)
            # plt.show()
            nx.drawing.nx_pydot.write_dot(G, "variant-graph.dot")
