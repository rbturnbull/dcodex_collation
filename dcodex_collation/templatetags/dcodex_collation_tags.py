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


# @register.filter
# def alignment_table_tbody(alignment):
#     html = "<tbody>"

#     is_rtl = alignment.is_rtl()
#     row_ids = alignment.row_ids()
#     for row_id in row_ids:
#         html += "<tr>"
#         html += "</tr>"
        

#             <tbody>
#                 {% for row in alignment.row_set.all %}
#                 <tr>
#                     {% if not alignment.is_rtl %}
#                         <th scope="row"><a href="/dcodex/ms/{{ row.transcription.manuscript.siglum }}/{{ row.transcription.verse.url_ref }}/" data-toggle="tooltip" data-placement="bottom" title="{{ row.transcription.manuscript }}">{{ row.transcription.manuscript.short_name }}</a></th>
#                     {% endif %}

#                     {% for cell in row.cell_set_display_order %}
#                         <td class="token {% if alignment.is_rtl %}rtl{%else%}ltr{%endif%}" data-row="{{ row.id }}" data-column="{{ cell.column.id }}" data-cell="{{ cell.id }}"  data-state="{{ cell.state.id }}" {% if cell.token %}draggable=true{% endif %}><div class="tokendiv" ><div class="shift shift-left"></div>{{ cell.token_display }}<div class="shift shift-right"></div></div></td>
#                     {% endfor %}

#                     {% if alignment.is_rtl %}
#                         <th scope="row"><a href="/dcodex/ms/{{ row.transcription.manuscript.siglum }}/{{ row.transcription.verse.url_ref }}/" data-toggle="tooltip" data-placement="bottom" title="{{ row.transcription.manuscript }}">{{ row.transcription.manuscript.short_name }}</a></th>
#                     {% endif %}

#                 </tr>
#                 {% endfor %}
                
#             </tbody>

#     <td class="token {% if alignment.is_rtl %}rtl{%else%}ltr{%endif%}" data-row="{{ row.id }}" data-column="{{ cell.column.id }}" data-cell="{{ cell.id }}"  data-state="{{ cell.state.id }}" {% if cell.token %}draggable=true{% endif %}><div class="tokendiv" ><div class="shift shift-left"></div>{{ cell.token_display }}<div class="shift shift-right"></div></div></td>
