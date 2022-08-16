import sys
from .models import Alignment, State, Row


def state_to_footnote(state, column, witnesses, is_primary:bool = False):
    text = state.str_at(column) if not is_primary else "txt"

    return f"{text} {witnesses}"

def write_latex(
    family, verses, witnesses=None, file=None, allow_ignore=True, primary_ms=None
):
    file = file or sys.stdout
    witnesses = witnesses or family.manuscripts()

    for verse in verses:
        alignment = Alignment.objects.filter(family=family, verse=verse).first()
        if not alignment:
            continue
        
        primary_row = Row.objects.filter(alignment=alignment, transcription__mansuscript=primary_ms) if primary_ms else None

        for column in alignment.column_set.all():
            
            state_witnesses = {
                state : [cell.row.transcription.manuscript.siglum for cell in state.cells_at(column)]
                for state in column.states(allow_ignore)
            }

            primary_state = column.atext
            if not primary_state:
                if primary_row:
                    primary_state = primary_row.state_at(column)
                else:

                cell = primary_row.
            states[0]
            primary_text = primary_state.text or "^"
            file.write(f"{primary_text}")
            if len(state_witnesses) > 1:
                footnotes = [state_to_footnote(state, witnesses, column) for state, witnesses in state_witnesses.items() if state.id != primary_state.id]
                footnotes.append( state_to_footnote(primary_state, column, is_primary=True) )

                file.write("\\footnote{" + " | ".join(footnotes) + "}\n")

            file.write(f" ")

            