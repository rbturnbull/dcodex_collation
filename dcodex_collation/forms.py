from django import forms
from dcodex.models import *

from .models import *

class ComparisonTableForm(forms.Form):
    manuscripts = forms.ModelMultipleChoiceField( queryset=Manuscript.objects.all(), widget=forms.CheckboxSelectMultiple, required=True )

    def comparison_table(self):
        manuscripts = self.cleaned_data['manuscripts']        
        comparison_array = np.zeros( (len(manuscripts), len(manuscripts)) )
        sigla = [ms.siglum for ms in manuscripts]

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

                comparison_array[index1,index2] = comparison_array[index2,index1] = agreement_count/total_count

        return sigla, comparison_array
