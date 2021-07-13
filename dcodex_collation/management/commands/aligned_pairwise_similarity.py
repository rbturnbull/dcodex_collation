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
        parser.add_argument('--window', type=int, default=15, help="The window size. Default: 15.")
        parser.add_argument('--full', type=str2bool, nargs='?', const=True, default=False, help="Restricts output to only sections where there enough alignments to fill the window.")
        parser.add_argument('--all-transitions', action='store_true', default=False, help="Forces the code to use all transition types. Default: Ignores transition types registered as TransitionTypeToIgnore.")
        parser.add_argument('--start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('--end', type=str, help="The ending verse of the passage selection. If this is not given, then it only uses the start verse.")
        parser.add_argument('--skip', type=str, nargs='+', help="A list of verses to skip.")


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

        # Use passage selection if given
        if options['start']:
            VerseClass = manuscript1.verse_class()
            assert  manuscript2.verse_class() == VerseClass

            start_verse_string = options['start'] or ""
            end_verse_string = options['end'] or ""

            selection_verse_ids = set( VerseClass.queryset_from_strings( start_verse_string, end_verse_string ).values_list( "id", flat=True ))
            intersection_verse_ids = intersection_verse_ids & selection_verse_ids

        # Skip verses if told to
        if options['skip']:
            verse_ids_to_skip = set([VerseClass.get_from_string(verse_ref_to_skip).id for verse_ref_to_skip in options['skip']])
            intersection_verse_ids = intersection_verse_ids - verse_ids_to_skip

        intersection_verse_ids_array = np.array( sorted(intersection_verse_ids) )

        states_ms1 = np.array(
            Cell.objects.filter( 
                row__alignment__verse__id__in=intersection_verse_ids, 
                row__transcription__manuscript=manuscript1,
            ).values_list( "state__id", flat=True )
        )
        states_ms2 = np.array(
            Cell.objects.filter( 
                row__alignment__verse__id__in=intersection_verse_ids, 
                row__transcription__manuscript=manuscript2,
            ).values_list( "state__id", flat=True )
        )
        verse_ids_per_cell = np.array(
            Cell.objects.filter( 
                row__alignment__verse__id__in=intersection_verse_ids, 
                row__transcription__manuscript=manuscript2,
            ).values_list( "row__alignment__verse__id", flat=True )
        )

        # Ignore transitions in transition types to ignore
        if options['all_transitions'] == False:
            for to_ignore in TransitionTypeToIgnore.objects.all():
                for x in range(10):
                    for transition in Transition.objects.filter( column__alignment__verse__id__in=intersection_verse_ids, transition_type=to_ignore.transition_type ):
                        
                        start_state_id = transition.start_state.id
                        end_state_id = transition.end_state.id

                        # print(states_ms2 == max(start_state_id, end_state_id))
                        # print(np.sum(states_ms2 == max(start_state_id, end_state_id)))
                        # print(np.sum(states_ms1 == max(start_state_id, end_state_id)))
                        np.place( states_ms2, states_ms2 == max(start_state_id, end_state_id), min(start_state_id, end_state_id) )
                        np.place( states_ms1, states_ms1 == max(start_state_id, end_state_id), min(start_state_id, end_state_id) )
                        
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


        ticks_count = min(100, verse_rank[-1]-verse_rank[0]+1 )
        xticks = np.linspace( verse_rank[0], verse_rank[-1], ticks_count )
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
