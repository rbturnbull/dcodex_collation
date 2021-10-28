from django.contrib import admin
from polymorphic.admin import (
    PolymorphicParentModelAdmin,
    PolymorphicChildModelAdmin,
    PolymorphicChildModelFilter,
)

from .models import *

# Register your models here.
@admin.register(Alignment)
class AlignmentAdmin(admin.ModelAdmin):
    raw_id_fields = ("verse",)


@admin.register(Row)
class RowAdmin(admin.ModelAdmin):
    raw_id_fields = ("transcription",)


@admin.register(Cell)
class CellAdmin(admin.ModelAdmin):
    raw_id_fields = (
        "token",
        "state",
    )


class TransitionClassifierChildAdmin(PolymorphicChildModelAdmin):
    """Base admin class for all child models"""

    base_model = TransitionClassifier


@admin.register(RegexTransitionClassifier)
class RegexTransitionClassifierAdmin(TransitionClassifierChildAdmin):
    base_model = RegexTransitionClassifier


@admin.register(TransitionClassifier)
class TransitionClassifierParentAdmin(PolymorphicParentModelAdmin):
    """The parent model admin"""

    base_model = TransitionClassifier  # Optional, explicitly set here.
    child_models = (RegexTransitionClassifier,)
    list_filter = (PolymorphicChildModelFilter,)  # This is optional.


class TransitionTypeInline(admin.TabularInline):
    model = TransitionType
    extra = 0


class TransitionRateInline(admin.TabularInline):
    model = TransitionRate
    extra = 0


@admin.register(TransitionRate)
class TransitionRateAdmin(admin.ModelAdmin):
    pass


@admin.register(RateSystem)
class RateSystemAdmin(admin.ModelAdmin):
    inlines = [
        TransitionRateInline,
    ]


admin.site.register(Column)
admin.site.register(TransitionType)
admin.site.register(Transition)
admin.site.register(State)
admin.site.register(Rate)

admin.site.register(Token)
admin.site.register(TransitionTypeToIgnore)
