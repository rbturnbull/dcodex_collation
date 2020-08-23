from django import template
import logging

register = template.Library()

@register.filter
def token(row, column):
    token_id = row.tokens[ column.order ]
    if token_id < 0:
        return ""
    return row.alignment.id_to_word[ token_id ]