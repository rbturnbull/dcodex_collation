from django import forms
from dcodex.models import *

from .models import *


class ComparisonTableForm(forms.Form):
    manuscripts = forms.ModelMultipleChoiceField(
        queryset=Manuscript.objects.all(),
        widget=forms.CheckboxSelectMultiple,
        required=True,
    )
    atext = forms.BooleanField(
        required=False, initial=False, label="Include the A-Text"
    )

    def comparison_table(self):
        manuscripts = self.cleaned_data["manuscripts"]
        sigla = [ms.siglum for ms in manuscripts]
        if self.cleaned_data["atext"]:
            sigla += ["A-Text"]

        comparison_array = calc_pairwise_comparison_array(
            manuscripts, atext=self.cleaned_data["atext"], raw=False,
        )

        return sigla, comparison_array
