from django import template
import logging

register = template.Library()

@register.filter
def token(row, column):
    return row.token_at( column )

@register.filter
def draggable(row, column):
    return "draggable=true" if row.token_id_at( column ) >= 0 else ""