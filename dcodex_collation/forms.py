from django import forms
from dcodex.models import *

from .models import *

class ComparisonTableForm(forms.Form):
    manuscripts = forms.ModelMultipleChoiceField( queryset=Manuscript.objects.all(), widget=forms.CheckboxSelectMultiple, required=True )

    def comparison_table(self):
        manuscripts = self.cleaned_data['manuscripts']        
        comparison_array = np.zeros( (len(manuscripts), len(manuscripts)) )
        sigla = [ms.siglum for ms in manuscripts]
        
        ignore_transition_type_ids = set(TransitionTypeToIgnore.objects.all().values_list('transition_type__id', flat=True))

        for index1, manuscript1 in enumerate(manuscripts):
            comparison_array[index1,index1] = 1.0
            for index2 in range( index1+1, len(manuscripts) ):
                manuscript2 = manuscripts[index2]

                column_ids_for_manuscript1 = Cell.objects.filter( row__transcription__manuscript=manuscript1 ).values_list('column__id', flat=True)
                column_ids_for_manuscript2 = Cell.objects.filter( row__transcription__manuscript=manuscript2 ).values_list('column__id', flat=True)

                column_ids_intersection = sorted(set(column_ids_for_manuscript1) & set(column_ids_for_manuscript2))

                total_count = len(column_ids_intersection)

                intersection_cells = Cell.objects.filter( column__id__in=column_ids_intersection )

                states_manuscript1 = intersection_cells.filter(row__transcription__manuscript=manuscript1 ).order_by('column__id').values_list( 'state__id', flat=True )
                states_manuscript2 = intersection_cells.filter(row__transcription__manuscript=manuscript2 ).order_by('column__id').values_list( 'state__id', flat=True )

                states_array_manuscript1 = np.array(list(states_manuscript1))
                states_array_manuscript2 = np.array(list(states_manuscript2))

                agreement_count = np.sum( states_array_manuscript1 == states_array_manuscript2 )

                column_ids_intersection_array = np.array(column_ids_intersection)
                disagreement_column_ids = column_ids_intersection_array[states_array_manuscript1 != states_array_manuscript2]

                disagreement_states_array_manuscript1 = states_array_manuscript1[ states_array_manuscript1 != states_array_manuscript2 ]
                disagreement_states_array_manuscript2 = states_array_manuscript2[ states_array_manuscript1 != states_array_manuscript2 ]

                for column_id, state1, state2 in zip( disagreement_column_ids, disagreement_states_array_manuscript1, disagreement_states_array_manuscript2 ):
                    column = Column.objects.get(id=column_id)
                    transition = Transition.objects.filter( column__id=column_id, start_state__id=state1, end_state__id=state2 ).first()
                    if not transition:
                        transition = Transition.objects.filter( column__id=column_id, start_state__id=state2, end_state__id=state1 ).first()
                        if transition:
                            transition = transition.create_inverse()
                    if transition:
                        # disagreement_transition_names.append( str(transition) )
                        if transition.transition_type.id in ignore_transition_type_ids:
                            agreement_count += 1

                comparison_array[index1,index2] = comparison_array[index2,index1] = agreement_count/total_count

        return sigla, comparison_array
