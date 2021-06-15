from django import template
from django.utils.safestring import mark_safe

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

@register.simple_tag
def array_2d_value(array, x, y):
    return "%.2f" % array[x,y]

@register.simple_tag
def array_2d_value_percent(array, x, y):
    return "%.2f%%" % (array[x,y] * 100.0)

@register.simple_tag
def row_cells_td(alignment, row):
    html = ""
    direction = "rtl" if alignment.is_rtl() else "ltr"
    values = row.cell_set_display_order().values_list(
        'id',
        'column__id',
        'state__id',
        'token__text',
    )
    for cell_id, column_id, state_id, token_text in values:
        if token_text == None:
            token_text = ""

        draggable = "draggable=true" if token_text else ""
        
        html += f'<td class="token {direction}" data-row="{row.id}" data-column="{column_id}" data-cell="{ cell_id }"  data-state="{state_id}" {draggable}><div class="tokendiv" ><div class="shift shift-left"></div>{ token_text }<div class="shift shift-right"></div></div></td>'
    return mark_safe(html)

@register.inclusion_tag('dcodex_collation/partials/_column_pair_row.html')
def column_pair_row(column, pair, pair_rank ):
    transition = column.transition_for_pair(pair_rank)

    return {"pair": pair, 'transition':transition}
