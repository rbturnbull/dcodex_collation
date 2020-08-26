from django import template
import logging

register = template.Library()

@register.filter
def token(row, column):
    return row.text_at( column )

@register.filter
def draggable(row, column):
    cell = row.cell_at(column)
    return "draggable=true" if cell.token else ""