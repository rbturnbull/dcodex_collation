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
    inlines = [TransitionRateInline,]



admin.site.register(Column)
admin.site.register(TransitionType)
admin.site.register(Transition)
admin.site.register(State)
admin.site.register(Rate)

admin.site.register(Token)
