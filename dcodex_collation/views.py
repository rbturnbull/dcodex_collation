import numpy as np

from django.shortcuts import render
from django.views.generic import DetailView, FormView, TemplateView, ListView
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404
from django.http import Http404
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required

from dcodex.models import *
from .models import *
from .forms import *


def manuscript_or_atext(siglum):
    manuscript = Manuscript.find(siglum)
    if not manuscript:
        if siglum.lower().replace("-", "") == "atext":
            manuscript = None
        else:
            raise Http404(f"Cannot find manuscript '{siglum}'")
    return manuscript


class AlignmentDetailView(LoginRequiredMixin, DetailView):
    model = Alignment
    template_name = "dcodex_collation/alignment.html"


class AlignmentForFamily(LoginRequiredMixin, TemplateView):
    template_name = "dcodex_collation/alignment.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)

        family_siglum = self.kwargs["family_siglum"]
        verse_ref = self.kwargs["verse_ref"]

        family = get_object_or_404(Family, name=family_siglum)
        verse = family.get_verse_from_string(verse_ref)

        alignment = Alignment.objects.filter(verse=verse, family=family).first()

        if not alignment:
            raise Http404(f"Alignment not found for verse {verse}.")
        #     gotoh_param = [6.6995597099885345, -0.9209875054657459, -5.097397327423096, -1.3005714416503906]
        #     alignment = align_family_at_verse( family, verse, gotoh_param )

        # return HttpResponse(str(family.id))
        next_alignment = Alignment.objects.filter(
            verse__rank__gt=verse.rank, family=family
        ).first()
        prev_alignment = (
            Alignment.objects.filter(verse__rank__lt=verse.rank, family=family)
            .order_by("-verse__rank")
            .first()
        )

        next_verse = next_alignment.verse if next_alignment else None
        prev_verse = prev_alignment.verse if prev_alignment else None

        context["alignment"] = alignment
        context["alignments_for_family"] = Alignment.objects.filter(family=family)
        context["next_verse"] = next_verse
        context["prev_verse"] = prev_verse

        return context


class AlignmentStatesForFamily(AlignmentForFamily):
    template_name = "dcodex_collation/alignment_states.html"


@login_required
def clear_empty(request):
    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    alignment.clear_empty()
    return HttpResponse("OK")


@login_required
def shift(request):

    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    row = get_object_or_404(Row, id=request.POST.get("row"))
    column = get_object_or_404(Column, id=request.POST.get("column"))
    delta = int(request.POST.get("delta"))
    if row.is_rtl():
        delta *= -1
    alignment.shift(row, column, delta)

    return HttpResponse("OK")


@login_required
def shift_to(request):
    alignment = get_object_or_404(Alignment, id=request.POST.get("alignment"))
    row = get_object_or_404(Row, id=request.POST.get("row"))
    start_column = get_object_or_404(Column, id=request.POST.get("start_column"))
    end_column = get_object_or_404(Column, id=request.POST.get("end_column"))

    if alignment.shift_to(row, start_column, end_column):
        return HttpResponse("OK")

    return HttpResponseBadRequest("Cannot shift to this column.")


@login_required
def classify_transition_for_pair(
    request, family_siglum, verse_ref, column_rank, pair_rank
):
    family = get_object_or_404(Family, name=family_siglum)
    verse = family.get_verse_from_string(verse_ref)
    alignment = get_object_or_404(Alignment, verse=verse, family=family)
    column = get_object_or_404(Column, alignment=alignment, order=column_rank)

    pairs = column.state_pairs()
    import logging

    logging.warning(pairs)
    if pair_rank >= len(pairs):
        column, pair_rank = column.next_pair(pair_rank)
        if column:
            pairs = column.state_pairs()

    if pair_rank is None:
        return HttpResponseBadRequest(
            f"Cannot find pair. Perhaps you have gone past the last location of variation."
        )
    pair = pairs[pair_rank]

    start_state = pair[0]
    end_state = pair[1]

    transition = Transition.objects.filter(
        column=column, start_state=start_state, end_state=end_state
    ).first()

    next_pair_url = column.next_pair_url(pair_rank)
    next_untagged_pair_url = column.next_untagged_pair_url(pair_rank)

    prev_pair_url = column.prev_pair_url(pair_rank)

    return render(
        request,
        "dcodex_collation/transition.html",
        context={
            "alignment": alignment,
            "column": column,
            "pair_rank": pair_rank,
            "start_state": start_state,
            "end_state": end_state,
            "transition": transition,
            "transition_types": TransitionType.objects.all(),
            "next_pair_url": next_pair_url,
            "next_untagged_pair_url": next_untagged_pair_url,
            "prev_pair_url": prev_pair_url,
            "alignments_for_family": Alignment.objects.filter(family=family),
        },
    )


@login_required
def set_transition_type(request):
    column = get_object_or_404(Column, id=request.POST.get("column"))
    transition_type = get_object_or_404(
        TransitionType, id=request.POST.get("transition_type")
    )
    inverse = request.POST.get("inverse")
    start_state = get_object_or_404(State, id=request.POST.get("start_state_id"))
    end_state = get_object_or_404(State, id=request.POST.get("end_state_id"))

    Transition.objects.update_or_create(
        column=column,
        start_state=start_state,
        end_state=end_state,
        defaults={
            "transition_type": transition_type,
            "inverse": inverse,
        },
    )
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
    # state = get_object_or_404(State, id=request.POST.get("state"))
    # if column.atext != state:
    #     HttpResponseBadRequest("Incorrect state for A-Text on this column.")

    column.atext_notes = request.POST.get("notes")
    column.save()

    return HttpResponse("OK")


@login_required
def pairwise_comparison(request, siglum1, siglum2):
    manuscript1 = manuscript_or_atext(siglum1)
    manuscript2 = manuscript_or_atext(siglum2)
    if manuscript1 is None:
        manuscript1, manuscript2 = manuscript2, manuscript1

    (
        agreement_count,
        total_count,
        disagreement_transitions,
    ) = find_disagreement_transitions(manuscript1, manuscript2)

    if manuscript2 is None:

        class ATextObject:
            def __str__(self):
                return "A Text"

            def siglum(self):
                return "A-Text"

        manuscript2 = ATextObject()

    context = dict(
        manuscript1=manuscript1,
        manuscript2=manuscript2,
        total_count=total_count,
        agreement_count=agreement_count,
        agreement_percentage=agreement_count / total_count * 100.0,
        disagreement_count=total_count - agreement_count,
        disagreement_transitions=disagreement_transitions,
    )

    return render(request, "dcodex_collation/pairwise_comparison.html", context=context)


@login_required
def disagreement_transitions_csv_view(request, siglum1, siglum2):
    manuscript1 = manuscript_or_atext(siglum1)
    manuscript2 = manuscript_or_atext(siglum2)
    if manuscript1 is None:
        manuscript1, manuscript2 = manuscript2, manuscript1

    response = HttpResponse(content_type="text/csv")
    response[
        "Content-Disposition"
    ] = f'attachment; filename="Disagreements-{siglum1}-{siglum2}.csv"'

    disagreements_transitions_csv(
        manuscript1=manuscript1, manuscript2=manuscript2, dest=response
    )

    return response


class ComparisonTableFormView(LoginRequiredMixin, FormView):
    template_name = "dcodex/form.html"
    form_class = ComparisonTableForm

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        sigla, comparison_array = form.comparison_table()

        context = dict(
            sigla=sigla,
            comparison_array=comparison_array,
        )
        return render(self.request, "dcodex_collation/comparison_table.html", context)


class ATextListView(LoginRequiredMixin, ListView):
    model = Column
    template_name = "dcodex_collation/atext_list.html"

    def get_queryset(self):
        return super().get_queryset().exclude(atext=None)


class TransitionTypeListView(LoginRequiredMixin, ListView):
    model = TransitionType
    template_name = "dcodex_collation/transitiontype_list.html"


class TransitionTypeDetailView(LoginRequiredMixin, DetailView):
    model = TransitionType
    template_name = "dcodex_collation/transitiontype_detail.html"


class ColumnDetailView(LoginRequiredMixin, DetailView):
    model = Column
    template_name = "dcodex_collation/column_detail.html"
    pk_url_kwarg = "order"

    def get_object(self, queryset=None):
        family_siglum = self.kwargs["family_siglum"]
        verse_ref = self.kwargs["verse_ref"]
        family = get_object_or_404(Family, name=family_siglum)
        verse = family.get_verse_from_string(verse_ref)
        return self.model.objects.get(
            alignment__verse=verse,
            alignment__family=family,
            order=self.kwargs["column_rank"],
        )
