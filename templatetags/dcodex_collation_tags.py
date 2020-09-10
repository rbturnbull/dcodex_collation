from django import template
import logging

register = template.Library()

@register.filter
def rows_with_state(column, state):
    return column.rows_with_state( state )

@register.filter
def token(row, column):
    return row.text_at( column )

@register.filter
def draggable(row, column):
    cell = row.cell_at(column)
    return "draggable=true" if cell and cell.token else ""

@register.filter
def state_str_at(state, column):
    return state.str_at(column)