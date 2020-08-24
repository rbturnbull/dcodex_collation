from django import template
import logging

register = template.Library()

@register.filter
def token(row, column):
    return row.token_at( column )