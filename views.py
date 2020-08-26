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

def classify_transition_for_pair(request, family_siglum, verse_ref, column_rank, pair_rank):
    verse = BibleVerse.get_from_string(verse_ref)
    family = get_object_or_404(Family, name=family_siglum)
    alignment = get_object_or_404(Alignment, verse=verse, family=family )
    column = get_object_or_404(Column, alignment=alignment, order=column_rank )

    pairs = column.state_pairs()
    if pair_rank >= len(pairs):
        return HttpResponseBadRequest(f"Rank for pair {pair_rank} too high.")
    pair = pairs[pair_rank]

    start_state = pair[0]
    end_state = pair[1]

    start_token_rows = column.rows_with_token_id( start_token_id )
    end_token_rows = column.rows_with_token_id( end_token_id )

    transition = Transition.objects.filter(column=column, start_token_id=start_token_id, end_token_id=end_token_id ).first()

    next_pair_url = column.next_pair_url( pair_rank )
    prev_pair_url = column.prev_pair_url( pair_rank )

    return render( request, "dcodex_collation/transition.html", context={
        'alignment':alignment,
        'column':column,
        'pair_rank':pair_rank,
        'start_token':alignment.id_to_word[pair[0]] if pair[0] != GAP else "OMIT",
        'end_token':alignment.id_to_word[pair[1]] if pair[1] != GAP else "OMIT",
        'start_token_id':start_token_id,
        'end_token_id':end_token_id,
        'transition':transition,
        'start_token_rows':start_token_rows,
        'end_token_rows':end_token_rows,
        'transition_types':TransitionType.objects.all(),
        'next_pair_url':next_pair_url,
        'prev_pair_url':prev_pair_url,
        })


def set_transition_type(request):
    column = get_object_or_404(Column, id=request.POST.get("column"))
    transition_type = get_object_or_404(TransitionType, id=request.POST.get("transition_type"))
    inverse = request.POST.get("inverse")
    start_token_id = request.POST.get("start_token_id")
    end_token_id = request.POST.get("end_token_id")

    Transition.objects.update_or_create( column=column, start_token_id=start_token_id, end_token_id=end_token_id, defaults={
        'transition_type': transition_type,
        'inverse': inverse,
    })
    return HttpResponse("OK")
