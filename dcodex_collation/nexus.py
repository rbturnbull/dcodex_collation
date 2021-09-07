import sys
from .models import * 

def write_nexus( family, verses, witnesses=None, file=None, allow_ignore=True, atext=False):
    file = file or sys.stdout
    witnesses = witnesses or family.manuscripts()

    witnesses_count = witnesses.count()
    if atext:
        witnesses_count += 1
    max_states = 0
    columns_count = 0
    for verse in verses:
        alignment = Alignment.objects.filter(family=family, verse=verse).first()
        if not alignment: continue

        for column in alignment.column_set.all():
            if column.only_punctuation():
                continue

            state_count = column.state_count(allow_ignore)
            max_states = max(max_states,state_count)    
            columns_count += 1

    file.write("#NEXUS\n")
    file.write("begin data;\n")
    file.write("\tdimensions ntax=%d nchar=%d;\n" % (witnesses_count, columns_count))
    file.write("\tformat datatype=Standard interleave=no gap=- missing=? ")    
    symbols =  "0123456789"
    max_states = len(symbols)
    file.write('symbols="')
    for x in range(max_states):
        file.write( "%s" % symbols[x] )
    file.write("\";\n")
    
    file.write('\tCHARSTATELABELS\n')
    index = 0
    for verse in verses:
        alignment = Alignment.objects.filter(family=family, verse=verse).first()
        if not alignment: continue
        for column in alignment.column_set.all():
            if column.only_punctuation():
                continue

            state_count = column.state_count(allow_ignore)       
            labels = ['State%d' % int(state) for state in range(state_count)]
            labels_joined = ". ".join( labels )
            column_name = f"{column.alignment.verse.url_ref()}:{column.order}"
            file.write(f"\t\t{index+1}  Column-{column_name} / {labels_joined}, \n")
            index += 1
    file.write("\t;\n")

    # Work out the longest length of a siglum for a witness to format the matrix
    max_siglum_length = 0
    for witness in witnesses:
        siglum = str(witness.short_name())
        if len(siglum) > max_siglum_length:
            max_siglum_length = len(siglum)
    
    margin = max_siglum_length + 5

    # Write the alignment matrix
    file.write("\tmatrix\n")

    if atext:
        siglum = "AText"
        file.write("\t%s%s" % (siglum, " "*(margin-len(siglum)) ))

        for verse in verses:
            alignment = Alignment.objects.filter(family=family, verse=verse).first()
            if not alignment: continue
            for column in alignment.column_set.all():
                if column.only_punctuation():
                    continue
                if not column.atext:
                    label = "?"
                else:
                    state_ids = [state.id for state in column.states(allow_ignore)]
                    atext_state = column.get_atext_state(allow_ignore)
                    label = str(state_ids.index(atext_state.id))

                file.write(label)
        file.write("\n")


    for witness in witnesses:
        siglum = str(witness.short_name())

        # Write the siglum and leave a gap for the margin
        file.write("\t%s%s" % (siglum, " "*(margin-len(siglum)) ))

        for verse in verses:
            #print(verse)
            alignment = Alignment.objects.filter(family=family, verse=verse).first()
            if not alignment: continue
            row = Row.objects.filter(transcription__manuscript=witness, alignment=alignment).first()
            

            for column in alignment.column_set.all():
                if column.only_punctuation():
                    continue
                if not row:
                    label = "?"
                else:
                    state_ids = [state.id for state in column.states(allow_ignore)]
                    # print("\n---------")
                    # print('state_ids', state_ids, column)
                    state = row.state_at(column, allow_ignore)
                    # print('state', state, state.id)

                    try:
                        label = str(state_ids.index(state.id)) if state else "?"
                    except Exception as err:
                        raise Exception(f"Cannot find label for column {column} row {row} verse {verse} witness {witness}.")


                file.write(label)

        file.write("\n")

    file.write('\t;\n')
    file.write('end;\n')