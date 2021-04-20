import numpy as np

from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

# import matplotlib.pyplot as plt
import plotly.graph_objects as go

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Command(BaseCommand):
    help = 'Shows a graph of the pairwise similarity of two manuscripts based on the same states in alignments.'

    def add_arguments(self, parser):
        parser.add_argument('siglum1', type=str, help="The siglum the first manuscript.")
        parser.add_argument('siglum2', type=str, help="The siglum the second manuscript.")
        parser.add_argument('--window', type=int, default=15, help="The window size.")
        parser.add_argument('--full', type=str2bool, nargs='?', const=True, default=False, help="Restricts output to only sections where there enough alignments to fill the window.")

    def handle(self, *args, **options):
        manuscript1 = Manuscript.find(options['siglum1'])
        manuscript2 = Manuscript.find(options['siglum2'])
        window = options['window']
        restrict_window_full = options['full']

        if window % 2 == 0:
            raise ValueError(f"Window value of {window} is not allowed. It must be odd.")

        half_window = (window - 1)/2

        print(f"{manuscript1 =}")
        print(f"{manuscript2 =}")

        verse_ids_manuscript1 = set( Row.objects.filter(transcription__manuscript=manuscript1).values_list( "alignment__verse__id", flat=True ))
        verse_ids_manuscript2 = set( Row.objects.filter(transcription__manuscript=manuscript2).values_list( "alignment__verse__id", flat=True ))
        print(f"{verse_ids_manuscript1 =}")
        print(f"{verse_ids_manuscript2 =}")        

        intersection_verse_ids = verse_ids_manuscript1 & verse_ids_manuscript2
        print(f"{intersection_verse_ids =}")

        intersection_verse_ids_array = np.array( sorted(intersection_verse_ids) )

        states_ms1 = np.array(
                Cell.objects.filter( 
                row__alignment__verse__id__in=intersection_verse_ids, 
                row__transcription__manuscript=manuscript1,
            ).values_list( "state_id", flat=True )
        )
        states_ms2 = np.array(
                Cell.objects.filter( 
                row__alignment__verse__id__in=intersection_verse_ids, 
                row__transcription__manuscript=manuscript2,
            ).values_list( "state_id", flat=True )
        )
        verse_ids_per_cell = np.array(
                Cell.objects.filter( 
                row__alignment__verse__id__in=intersection_verse_ids, 
                row__transcription__manuscript=manuscript2,
            ).values_list( "row__alignment__verse__id", flat=True )
        )

        verse_class = manuscript1.verse_class()

        # This should be optimised
        column_counts = np.array([Column.objects.filter(alignment__verse__id=verse_id).count() for verse_id in intersection_verse_ids_array])

        agreements = (states_ms1 == states_ms2)

        rolling_similarity = []
        for verse_id in intersection_verse_ids_array:
            indexes_in_window = (intersection_verse_ids_array >= verse_id - half_window) & (intersection_verse_ids_array <= verse_id + half_window)
            verses_in_window = np.sum(indexes_in_window)

            if restrict_window_full and verses_in_window != window:
                rolling_similarity.append(None)
                continue

            count = np.sum(column_counts[ indexes_in_window ])
            agreement_count = np.sum(agreements[ (verse_ids_per_cell >= verse_id - half_window) & (verse_ids_per_cell <= verse_id + half_window) ])
            rolling_similarity.append(agreement_count/count*100)

        rolling_similarity = np.array(rolling_similarity)

        print(states_ms1.shape)
        print(np.min(states_ms1))
        print(states_ms2.shape)
        print(np.min(states_ms2))
        print(f"{verse_ids_per_cell =}")
        print(f"{verse_ids_per_cell.shape =}")

        print(f"{intersection_verse_ids_array =}")
        print(f"{column_counts =}")
        print(f"{agreements =}")
        print(f"{rolling_similarity =}")
        print(f"{rolling_similarity.shape =}")

        verses = [verse_class.objects.get(id=verse_id) for verse_id in intersection_verse_ids_array]
        verse_refs = [verse.reference(abbreviation=True) for verse in verses]
        verse_rank = [verse.rank for verse in verses]
        
        # plt.plot(intersection_verse_ids_array, rolling_similarity, '-o')

        # # plt.scatter(intersection_verse_ids_array, rolling_similarity)
        # plt.show()



        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=verse_rank, 
                y=rolling_similarity,
                text=verse_refs,
                mode='lines+markers',
                name='Rolling Similarity'
            )
        )

        xticks = np.linspace( verse_rank[0], verse_rank[-1], 100 )
        xtick_verses = [verse_class.objects.filter(rank=rank).first() for rank in xticks]
        tick_text = [ verse.reference(abbreviation=True) if verse else "" for verse in xtick_verses]

        fig.update_layout(
            xaxis = dict(
                tickmode='array',
                tickvals=xticks,
                ticktext=tick_text,
            )
        )

        fig.show()
