from django import forms
from dcodex.models import *

from .models import *

class ComparisonTableForm(forms.Form):
    manuscripts = forms.ModelMultipleChoiceField( queryset=Manuscript.objects.all(), widget=forms.CheckboxSelectMultiple, required=True )

    def comparison_table(self):
        manuscripts = self.cleaned_data['manuscripts']        
        sigla = [ms.siglum for ms in manuscripts]

        comparison_array = calc_pairwise_comparison_array(manuscripts)        

        return sigla, comparison_array
