from django.contrib import admin

from .models import *

# Register your models here.
@admin.register(Alignment)
class AlignmentAdmin(admin.ModelAdmin):
    raw_id_fields = ("verse",)    


@admin.register(Row)
class Row(admin.ModelAdmin):
    raw_id_fields = ("transcription",)    


admin.site.register(Column)
admin.site.register(TransitionType)
admin.site.register(Transition)
admin.site.register(Cell)
admin.site.register(State)
