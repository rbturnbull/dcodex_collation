from django.shortcuts import render
from django.views.generic.detail import DetailView
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404

from .models import *
from dcodex.models import *
from dcodex_bible.models import *

class AlignmentDetailView(DetailView):
    model = Alignment
    template_name = "dcodex_collation/alignment.html"


def alignment_for_family(request, family_siglum, verse_ref):
    verse = BibleVerse.get_from_string(verse_ref)
    family = get_object_or_404(Family, name=family_siglum)

    alignment = Alignment.objects.filter( verse=verse, family=family ).first()
    if not alignment:
        gotoh_param = [6.6995597099885345, -0.9209875054657459, -5.097397327423096, -1.3005714416503906]
        alignment = align_family_at_verse( family, verse, gotoh_param )    

    #return HttpResponse(str(family.id))
    return render( request, "dcodex_collation/alignment.html", context={'alignment':alignment})

def shift(request):

    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    row = get_object_or_404(Row, id=request.POST.get("row"))
    column = get_object_or_404(Column, id=request.POST.get("column"))
    delta = int(request.POST.get("delta"))
    maxshift = int(request.POST.get("maxshift"))

    if maxshift:
        alignment.maxshift( row, column, delta )    
    else:
        alignment.shift( row, column, delta )

    return HttpResponse("OK")


def shift_to(request):
    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    row = get_object_or_404(Row, id=request.POST.get("row"))
    start_column = get_object_or_404(Column, id=request.POST.get("start_column"))
    end_column = get_object_or_404(Column, id=request.POST.get("end_column"))

    if alignment.shift_to( row, start_column, end_column ):
        return HttpResponse("OK")
    
    return HttpResponseBadRequest("Cannot shift to this column.")
