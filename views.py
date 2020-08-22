from django.shortcuts import render
from django.views.generic.detail import DetailView

from .models import *

class AlignmentDetailView(DetailView):
    model = Alignment
    template_name = "dcodex_collation/alignment.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

