from django.contrib import admin

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
    raw_id_fields = ("token","state",)    


admin.site.register(Column)
admin.site.register(TransitionType)
admin.site.register(Transition)
admin.site.register(State)

admin.site.register(TransitionRate)
admin.site.register(Rate)
admin.site.register(RateSystem)
admin.site.register(Token)
