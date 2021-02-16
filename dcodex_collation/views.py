import numpy as np

from django.shortcuts import render
from django.views.generic import DetailView, FormView
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404
from django.http import Http404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required

from dcodex.models import *
from .models import *
from .forms import *



class AlignmentDetailView(LoginRequiredMixin, DetailView):
    model = Alignment
    template_name = "dcodex_collation/alignment.html"

@login_required
def alignment_for_family(request, family_siglum, verse_ref):
    family = get_object_or_404(Family, name=family_siglum)
    verse = family.get_verse_from_string( verse_ref )

    alignment = Alignment.objects.filter( verse=verse, family=family ).first()
    
    if not alignment:
        raise Http404(f"Alignment not found for verse {verse}.")
    #     gotoh_param = [6.6995597099885345, -0.9209875054657459, -5.097397327423096, -1.3005714416503906]
    #     alignment = align_family_at_verse( family, verse, gotoh_param )    

    #return HttpResponse(str(family.id))
    next_alignment = Alignment.objects.filter( verse__rank__gt=verse.rank, family=family ).first()
    prev_alignment = Alignment.objects.filter( verse__rank__lt=verse.rank, family=family ).order_by('-verse__rank').first()

    context = {
        'alignment':alignment, 
        'alignments_for_family': Alignment.objects.filter( family=family ),
        'next_verse': next_alignment.verse,
        'prev_verse': prev_alignment.verse,
    }

    return render( request, "dcodex_collation/alignment.html", context=context )

@login_required
def clear_empty(request):
    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    alignment.clear_empty( )
    return HttpResponse("OK")

@login_required
def shift(request):

    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    row = get_object_or_404(Row, id=request.POST.get("row"))
    column = get_object_or_404(Column, id=request.POST.get("column"))
    delta = int(request.POST.get("delta"))
    if row.is_rtl():
        delta *= -1
    alignment.shift( row, column, delta )

    return HttpResponse("OK")

@login_required
def shift_to(request):
    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    row = get_object_or_404(Row, id=request.POST.get("row"))
    start_column = get_object_or_404(Column, id=request.POST.get("start_column"))
    end_column = get_object_or_404(Column, id=request.POST.get("end_column"))

    if alignment.shift_to( row, start_column, end_column ):
        return HttpResponse("OK")
    
    return HttpResponseBadRequest("Cannot shift to this column.")

@login_required
def classify_transition_for_pair(request, family_siglum, verse_ref, column_rank, pair_rank):
    family = get_object_or_404(Family, name=family_siglum)
    verse = family.get_verse_from_string( verse_ref )
    alignment = get_object_or_404(Alignment, verse=verse, family=family )
    column = get_object_or_404(Column, alignment=alignment, order=column_rank )

    pairs = column.state_pairs()
    import logging
    logging.warning(pairs)
    if pair_rank >= len(pairs):
        column, pair_rank = column.next_pair(pair_rank)
        pairs = column.state_pairs()

    if pair_rank is None:
        return HttpResponseBadRequest(f"Cannot find pair.")
    pair = pairs[pair_rank]

    start_state = pair[0]
    end_state = pair[1]

    transition = Transition.objects.filter(column=column, start_state=start_state, end_state=end_state ).first()


    next_pair_url = column.next_pair_url( pair_rank )
    next_untagged_pair_url = column.next_untagged_pair_url( pair_rank )

    prev_pair_url = column.prev_pair_url( pair_rank )

    return render( request, "dcodex_collation/transition.html", context={
        'alignment':alignment,
        'column':column,
        'pair_rank':pair_rank,
        'start_state':start_state,
        'end_state':end_state,
        'transition':transition,
        'transition_types':TransitionType.objects.all(),
        'next_pair_url':next_pair_url,
        'next_untagged_pair_url':next_untagged_pair_url,
        'prev_pair_url':prev_pair_url,
        'alignments_for_family': Alignment.objects.filter( family=family ),
        })

@login_required
def set_transition_type(request):
    column = get_object_or_404(Column, id=request.POST.get("column"))
    transition_type = get_object_or_404(TransitionType, id=request.POST.get("transition_type"))
    inverse = request.POST.get("inverse")
    start_state = get_object_or_404(State, id=request.POST.get("start_state_id"))
    end_state = get_object_or_404(State, id=request.POST.get("end_state_id"))

    Transition.objects.update_or_create( column=column, start_state=start_state, end_state=end_state, defaults={
        'transition_type': transition_type,
        'inverse': inverse,
    })
    return HttpResponse("OK")

@login_required
def set_atext(request):
    column = get_object_or_404(Column, id=request.POST.get("column"))
    state = get_object_or_404(State, id=request.POST.get("state"))
    column.atext = state
    column.save()
    return HttpResponse("OK")

@login_required
def remove_atext(request):
    column = get_object_or_404(Column, id=request.POST.get("column"))
    state = get_object_or_404(State, id=request.POST.get("state"))
    if column.atext != state:
        HttpResponseBadRequest("Incorrect state for A-Text on this column.")

    column.atext = None
    column.save()

    return HttpResponse("OK")    

@login_required
def save_atext_notes(request):
    column = get_object_or_404(Column, id=request.POST.get("column"))
    state = get_object_or_404(State, id=request.POST.get("state"))
    if column.atext != state:
        HttpResponseBadRequest("Incorrect state for A-Text on this column.")

    column.atext_notes = request.POST.get("notes")
    column.save()

    return HttpResponse("OK")    


@login_required
def pairwise_comparison(request, siglum1, siglum2):
    manuscript1 = Manuscript.find(siglum1)
    if not manuscript1:
        raise Http404(f"Cannot find manuscript '{siglum1}'")
    manuscript2 = Manuscript.find(siglum2)
    if not manuscript2:
        raise Http404(f"Cannot find manuscript '{siglum2}'")


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


    # disagreement_transition_names = []
    disagreement_transitions = []

    for column_id, state1, state2 in zip( disagreement_column_ids, disagreement_states_array_manuscript1, disagreement_states_array_manuscript2 ):
        column = Column.objects.get(id=column_id)
        transition = Transition.objects.filter( column__id=column_id, start_state__id=state1, end_state__id=state2 ).first()
        if not transition:
            transition = Transition.objects.filter( column__id=column_id, start_state__id=state2, end_state__id=state1 ).first()
            if transition:
                transition = transition.create_inverse()
        if transition:
            # disagreement_transition_names.append( str(transition) )
            disagreement_transitions.append( transition )

    context = dict(
        manuscript1=manuscript1,
        manuscript2=manuscript2,
        total_count=total_count,
        agreement_count=agreement_count,
        agreement_percentage=agreement_count/total_count*100.0,
        disagreement_count=total_count-agreement_count,
        disagreement_transitions=disagreement_transitions,
    )

    return render( request, "dcodex_collation/pairwise_comparison.html", context=context)


class ComparisonTableFormView(LoginRequiredMixin, FormView):
    template_name = 'dcodex/form.html'
    form_class = ComparisonTableForm

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        sigla, comparison_array = form.comparison_table()

        context = dict(
            sigla=sigla,
            comparison_array=comparison_array,
        )
        return render(self.request, 'dcodex_collation/comparison_table.html', context)


