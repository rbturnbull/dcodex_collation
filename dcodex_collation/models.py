from django.db import models
from scipy.cluster import hierarchy
import gotoh
import numpy as np
# from jsonfield import JSONField
# from ndarray import NDArrayField
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from dcodex.models import *
from dcodex.models.markup import *
from django.urls import reverse
from django.db.models import F

GAP = -1

def get_gap_state():
    return State.objects.filter(text=None).first()

def tokenize_strings( transcriptions ):
    return [transcription.tokenize() for transcription in transcriptions]


def update_alignment( alignment, **kwargs ):
    for row in alignment.row_set.all():
        update_transcription_in_alignment( row.transcription, alignment=alignment, **kwargs )


def update_transcription_in_alignment( transcription, gotoh_param, alignment=None, gap_open=-5, gap_extend=-2 ):

    if alignment == None:
        alignment = Alignment.objects.get(row__transcription=transcription)

    token_strings = tokenize_strings([transcription])[0]
    tokens = []
    for token_text in token_strings:
        token = alignment.token_set.filter()
        token, _ = Token.objects.update_or_create( alignment=alignment, text=token_text, defaults={
            "regularized": normalize_transcription(token_text),
        })
        tokens.append(token)
    token_ids = [token.id for token in tokens]

    current_row = alignment.row_set.get(transcription__manuscript=transcription.manuscript)
    current_tokens = list(current_row.cell_set.exclude(token=None).values_list("token__id", flat=True))

    # Check to see if the tokens are identical, if so, then we don't need to update this row
    if current_tokens == token_ids:
        print(f"Transcription for {transcription.manuscript} is up-to-date.")
        return


    # Create Scoring Matrix
    all_tokens = Token.objects.filter(alignment=alignment)
    all_token_ids = all_tokens.values_list("id", flat=True)
    all_token_regularized = all_tokens.values_list("regularized", flat=True)
    token_id_to_index = {k: v for v, k in enumerate(all_token_ids)}

    n = len(all_token_ids)
    scoring_matrix = np.zeros( (n,n), dtype=np.float32 )
    for index_i in range(n):
        token_i = all_token_regularized[index_i]
        for index_j in range(index_i+1):
            token_j = all_token_regularized[index_j]
            scoring_matrix[index_i,index_j] = scoring_matrix[index_j,index_i] = gotoh.score( token_i, token_j, *gotoh_param )

    # print(f"{scoring_matrix =}")
    # print(f"{gotoh_param =}")
    # return

    # Create alignment array from existing alignment
    rows = alignment.row_set.all()
    alignment_array = np.zeros( (alignment.column_set.count(), rows.count()), dtype=np.int )
    for row_index, row in enumerate(rows):
        row_token_ids = row.cell_set.all().values_list("token__id", flat=True)
        alignment_array[:,row_index] = np.asarray( [token_id_to_index[token_id] if token_id != None else GAP for token_id in row_token_ids] )

    # Create alignment array for current transcription
    current_transcription_indexes = [ token_id_to_index[token_id] for token_id in token_ids]
    current_transcription_as_alignment = np.expand_dims( np.asarray(current_transcription_indexes, dtype=np.int ), axis=1)

    # Run MSA
    pointers = gotoh.pointers( alignment_array, current_transcription_as_alignment, matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend, show_score=False )
    UP, LEFT, DIAG, NONE = gotoh.pointer_constants()

    # print(pointers)
    # print(gotoh.gotoh.pointers_ascii(pointers))
    # raise Exception("show")


    # print('current_transcription_as_alignment', current_transcription_as_alignment)
    # print('current_transcription_as_alignment.shape', current_transcription_as_alignment.shape)
    all_token_ids = list(all_token_ids)
    all_token_regularized = list(all_token_regularized)
    # for x in range(current_transcription_as_alignment.shape[0]):
    #     print( current_transcription_as_alignment[x,0], all_token_ids[ current_transcription_as_alignment[x,0] ], all_token_regularized[ current_transcription_as_alignment[x,0] ] )

    # print(alignment.ascii())

    # print('xxxxx')

    # print(alignment_array)
    for x in range(alignment_array.shape[0]):
        for y in range(alignment_array.shape[1]):
            regularized_string = "-" if alignment_array[x,y] == GAP else all_token_regularized[ alignment_array[x,y] ]
            token_id = "-" if alignment_array[x,y] == GAP else all_token_ids[ alignment_array[x,y] ]
            print( alignment_array[x,y], token_id, regularized_string, end="\t" )
        print()



    # for column in alignment.column_set.all():
    #     print(column.order, 'column.order')


    # Remove row for out-of-date transcription
    if current_row:
        current_row.delete()

    # Create new row for this transcription
    current_row = Row( alignment=alignment, transcription=transcription )
    current_row.save()

    # print(alignment.ascii())

    # print(f"{alignment_array.shape =}")
    # print(f"{current_transcription_as_alignment.shape =}")

    flip = 0
    max_j = alignment_array.shape[0]
    max_i = current_transcription_as_alignment.shape[0]

    if max_j > max_i:
        flip = 1
        max_i, max_j = max_j, max_i    
        print("FLIP")

    # Go through pointers and add gaps as necessary
    seqlen = max_i + max_j
    alignment_index = seqlen - 1
    i = max_i
    j = max_j
    p = pointers[i, j]

    pointer_code = {1: "UP", 2:"LEFT", 3: "DIAG", 4: "NONE" }

    gap_state = get_gap_state()

    # print(f"{max_i =}")
    # print(f"{max_j =}")

    while p != NONE:
        # Adjust indexes
        if p == DIAG:
            i -= 1
            j -= 1
        elif p == LEFT:
            j -= 1
        elif p == UP:
            i -= 1

        print(flip, p, pointer_code[p], ", flip, p")
        if (p == LEFT and not flip) or (p == UP and flip) or (p == DIAG):
            # Rerank the column
            column_rank = i if flip else j
            # print(f"looking for column {column_rank}")
            column = alignment.column_set.filter(order=column_rank).first()

            column.order = alignment_index + seqlen
            # print(f"\tcolumn {column_rank} going to {column.order}")

            column.save()
        else:
            # Add Gap
            column = Column(alignment=alignment, order=alignment_index + seqlen)
            column.save()
            for row in alignment.row_set.exclude(id=current_row.id):
                cell = Cell(row=row, column=column, state=gap_state, token=None)
                cell.save()

        # print(column.order, 'column.order --------------- ')

        if (p == LEFT and flip) or (p == UP and not flip) or (p == DIAG):
            current_row_rank = j if flip else i
            # print(f"{i =}")
            # print(f"{j =}")
            # print(f"{current_row_rank =}")
            # print(f"{current_transcription_indexes =}")
            index = current_transcription_indexes[current_row_rank]
            regularized = all_token_regularized[index]
            state, _ = State.objects.update_or_create( text=regularized )
            token = all_tokens[index]
            cell = Cell(row=current_row, column=column, state=state, token=token)
            cell.save()
        else:
            cell = Cell(row=current_row, column=column, state=gap_state, token=None)
            cell.save()

        alignment_index -= 1
        p = pointers[i, j]



    # Update column ranks
    # for column in alignment.column_set.all():
    #     print(column.order, 'column.order')
    # print("-------")
    alignment.column_set.update(order=F('order') - 1 - alignment_index - seqlen)
    # for column in alignment.column_set.all():
    #     print(column.order, 'column.order')

    # print(alignment.ascii())

    return alignment


def align_family_at_verse(family, verse, gotoh_param, iterations_count = 1, gap_open=-5, gap_extend=-2, exclude_empty=False):
    transcriptions = list(family.transcriptions_at(verse))
    transcriptions = [t for t in transcriptions if '_' not in t.verse.url_ref()] # hack for certain lectionaries
    
    # Exclude transcriptions with just empty strings. If a verse is omitted, then there should be some markup in the transcription
    if exclude_empty:
        transcriptions = [t for t in transcriptions if len(t.transcription.strip()) > 0]

    if len(transcriptions) < 2:
        print(f"Too few transcriptions at {verse}")
        return


    # Distance matrix
    distance_matrix_as_vector = []
    for x_index, x in enumerate(transcriptions):
        x_string = x.normalize()
        for y_index, y in enumerate(transcriptions):
            if y_index >= x_index:
                break
            y_string = y.normalize()
            distance = gotoh.nonmatches( x_string, y_string, *gotoh_param )
            distance_matrix_as_vector.append( distance )

    # Guide tree
    method="average"
    linkage = hierarchy.linkage(distance_matrix_as_vector, method)

    # Tokenize
    token_strings = tokenize_strings(transcriptions)
    print(token_strings)

    vocab = {}
    vocab_index = 0
    alignments = []
    for transcription_token_strings in token_strings:
        transcription_tokens = []
        for token_string in transcription_token_strings:
            if token_string not in vocab:
                vocab[token_string] = vocab_index
                vocab_index += 1
            transcription_tokens.append(vocab[token_string])

        alignment = np.expand_dims( np.asarray(transcription_tokens, dtype=np.int ), axis=1)
        #print(alignment)
        alignments.append( alignment )

    id_to_word = np.asarray(list(vocab.keys()) )

    # Scoring Matrix
    n = len(id_to_word)
    scoring_matrix = np.zeros( (n,n), dtype=np.float32 )
    for index_i in range(n):
        token_i = normalize_transcription(str(id_to_word[index_i]))
        for index_j in range(index_i+1):
            token_j = normalize_transcription(str(id_to_word[index_j]))
            #print(index_i,index_j)
            scoring_matrix[index_i,index_j] = gotoh.score( token_i, token_j, *gotoh_param )
    #print(scoring_matrix)
    #print([scoring_matrix[1][2]])
    #print([scoring_matrix[2][1]])
    #return

    # Initial Progressive Alignment
    alignment_transcriptions = [[transcription] for transcription in transcriptions]


    for link_index, link_row in enumerate(linkage):
        node_id = len(transcriptions) + link_index
        left_node_id = int(link_row[0])
        right_node_id = int(link_row[1])

        left_alignment = alignments[left_node_id]
        right_alignment = alignments[right_node_id]
        # print("--")
        # print('left sigla', [t.manuscript.siglum for t in alignment_transcriptions[left_node_id]], left_alignment)
        # print('right sigla', [t.manuscript.siglum for t in alignment_transcriptions[right_node_id]], right_alignment)
# #        return 

        # if left_alignment == None or right_alignment == None:
        #     raise Exception(f"One of the alignments is null. L: {left_alignment}. R: {right_alignment}")

        new_alignment = gotoh.msa( left_alignment, right_alignment, matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend )
        # print(f"new_alignment = {new_alignment}")
        alignments.append( new_alignment )
        alignment_transcriptions.append(  alignment_transcriptions[left_node_id] + alignment_transcriptions[right_node_id] )
        # print(f"left_alignment.shape = {left_alignment.shape}")
        # print(f"right_alignment.shape = {right_alignment.shape}")
        # print(f"new_alignment.shape = {new_alignment.shape}")

        #
        # if link_index == 6:
        #    break

    alignment_array = alignments[-1]

    alignment_transcriptions = alignment_transcriptions[-1]

    for iteration in range(iterations_count):
        for transcription in alignment_transcriptions:
            #print(transcription.manuscript.siglum)
            #print(alignment_array)
            # Pop row
            row = alignment_array[:,0]
            row = np.delete( row, row < 0 ) # Remove all the gaps
            #print(row)
            row = np.expand_dims( row, axis=1)
            alignment_array = alignment_array[:,1:]
            # Realign row

            alignment_array = gotoh.msa( alignment_array, row, matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend, visualize=False )
            #print(alignment_array)
            #if transcription.manuscript.siglum == "J67_esk":
            #    return

    #print("alignment", alignment.shape)
    #print("len(alignment_transcriptions)", len(alignment_transcriptions))

    #print(alignment_transcriptions)
    


    alignment, _ = Alignment.objects.update_or_create( family=family, verse=verse )
    id_to_word = list(vocab.keys())
    for index, token_text in enumerate(id_to_word):
        Token.objects.update_or_create( alignment=alignment, text=token_text, defaults={
            "regularized": normalize_transcription(token_text),
            "rank": index,
        })

    print(alignment_array)

    
    
    for order in range( alignment_array.shape[0] ):
        column, _ = Column.objects.update_or_create( order=order, alignment=alignment, defaults={} )
    Column.objects.filter(alignment=alignment, order__gte=alignment_array.shape[0] ).delete()

    Row.objects.filter(alignment=alignment).delete()
    for transcription, tokens in zip( alignment_transcriptions, np.rollaxis(alignment_array, 1) ):
        row, _ = Row.objects.update_or_create( transcription=transcription, alignment=alignment )
        print(f"row: {row.id}. Transcription: {transcription.id}. manuscript {transcription.manuscript}")
        row.create_cells_for_tokens(tokens, id_to_word)    
            #print(cell, row.transcription, column.order, token, state)

    assert len(alignment_transcriptions) == alignment_array.shape[1]

    #dn = hierarchy.dendrogram(linkage, orientation='right',labels=[transcription.manuscript.siglum for transcription in transcriptions])
    #plt.show()
    return alignment


    

class Alignment(models.Model):
    family = models.ForeignKey( Family, on_delete=models.SET_DEFAULT, default=None, null=True, blank=True )
    verse = models.ForeignKey( Verse, on_delete=models.CASCADE )

    class Meta:
        ordering = ['family', 'verse']

    def __str__(self):
        return f"{self.family} - {self.verse}"

    def ascii(self):
        string = ""

        rows = self.row_set.all()
        max_siglum_len = 0
        for row in rows:
            max_siglum_len = max(max_siglum_len, len(row.transcription.manuscript.short_name()))

        d = " | "
        for row in rows:
            siglum = row.transcription.manuscript.short_name()
            string += siglum + (" " * (max_siglum_len - len(siglum)))

            for cell in row.cell_set.all():
                token_string = "–" if not cell.token else cell.token.text
                max_token_len = cell.column.max_token_len()
                string += d + token_string + (" " * (max_token_len - len(token_string)))

            string += "\n"

        return string 

    def get_absolute_url(self):
        return reverse("alignment_for_family", kwargs={"family_siglum": self.family.name, "verse_ref": self.verse.url_ref() })

    def is_rtl(self):
        row = self.row_set.first()
        return row.is_rtl()

    def column_set_display_order(self):
        if self.is_rtl():
            return self.column_set.all().reverse()
        return self.column_set.all()

    def add_column(self, new_column_order):
        columns = self.column_set.filter( order__gte=new_column_order )
        for c in columns:
            c.order += 1
            c.save()
        column = Column(alignment=self, order=new_column_order)
        column.save()

        gap_state = get_gap_state()
        for row in self.row_set.all():
            Cell( row=row, column=column, state=gap_state, token=None).save()

        return column

    def empty_columns(self):
        return np.asarray( [column.is_empty() for column in self.column_set.all()] )

    def clear_empty(self):
        to_delete = []
        for column in self.column_set.all().reverse():
            if column.is_empty():
                # import logging
                # logging.warning("column empty"+ str(column.order))
                for c in list(self.column_set.filter(order__gt=column.order)):
                    #logging.warning("shifting " + str(c.order))
                    #continue

                    c.order -= 1
                    c.save()
                to_delete.append( column.id )
        Column.objects.filter(id__in=to_delete).delete()

    def shift_to( self, row, start_column, end_column ):
        import logging
        if start_column.order == end_column.order:
            logging.warning("columns equal")
            return False

        start_cell = row.cell_at(start_column)
        if start_cell.token == None:
            logging.warning("cannot find start cell")
            return False

        delta = -1 if start_column.order < end_column.order else 1

        # Ensure that there are all gaps between the two columns
        if start_column.order < end_column.order:
            intermediate = row.cell_set.filter( column__order__gt=start_column.order, column__order__lte=end_column.order )
        else:    
            intermediate = row.cell_set.filter( column__order__lt=start_column.order, column__order__gte=end_column.order )
        
        if intermediate.exclude(token=None).count() > 0:
            logging.warning("intermediate cells not empty")
            return False

        target_cell = row.cell_at(end_column)
        target_cell.token = start_cell.token
        target_cell.state = start_cell.state
        target_cell.save()

        start_cell.state = get_gap_state()
        start_cell.token = None
        start_cell.save()
        
        return True

    def shift(self, row, column, delta):
        # if the target column is empty, then just transfer over
        target_cell = Cell.objects.get(column__order=column.order + delta, row=row)
        if target_cell.token:
            # if the next target column is full, then create new column
            new_column_order = column.order + delta if delta > 0 else column.order
            new_column = self.add_column( new_column_order )

            # Get the objects from database again because the values have changed.
            target_cell = Cell.objects.get(column=new_column, row=row)
            column = Column.objects.get(id=column.id)
        

        start_cell = row.cell_at(column)
        target_cell.token = start_cell.token
        target_cell.state = start_cell.state
        target_cell.save()

        start_cell.state = get_gap_state()
        start_cell.token = None
        start_cell.save()
        
        # Check that no columns are empty
        #self.clear_empty( )

    def row_ids(self):
        return self.row_set.values_list('id', flat=True)

class Row(models.Model):
    transcription = models.ForeignKey( VerseTranscription, on_delete=models.CASCADE )
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    # tokens = NDArrayField(help_text="Numpy array for the tokens. IDs correspond to the vocab in the alignment", blank=True, null=True)

    class Meta:
        ordering = ["transcription__manuscript",]

    def is_rtl(self):
        return self.transcription.manuscript.text_direction == TextDirection.RIGHT_TO_LEFT

    def create_cells_for_tokens(self, tokens, id_to_word):
        for rank, token_id in enumerate(tokens):
            column, _ = Column.objects.get_or_create(alignment=self.alignment, order=rank)

            #print(column)
            if token_id == -1:
                token = None
            else:
                token_text = id_to_word[token_id]
                token, _ = Token.objects.get_or_create(text=token_text, alignment=self.alignment, defaults=dict(rank=token_id),)

            # Create State
            if token and "⧙" in token.text:
                state = None
            else:
                text = token.regularized if token else None
                state, _ = State.objects.update_or_create( text=text )

            # Create Cell
            cell, _ = Cell.objects.update_or_create( row=self, column=column, defaults={
                'token':token,
                "state":state,
            })            

    def cell_set_display_order(self):
        if self.is_rtl():
            return self.cell_set.all().reverse()
        return self.cell_set.all()        

    def token_id_at( self, column ): # should this be token_at ?
        cell = self.cell_at(column)
        if cell:
            return cell.token

    # def token_at( self, column ):
    #     token_id = self.token_id_at( column )
    #     if token_id < 0:
    #         return ""
    #     return self.alignment.id_to_word[ token_id ]        

    def cell_at(self, column):
        return self.cell_set.filter(column=column).first()

    def text_at(self, column):
        #return str(column.order)
        cell = self.cell_at(column)
        if not cell:
            return "ERROR"
        if cell.token and cell.token.text:
            return cell.token.text
        return ""

    def state_at(self,column):
        cell = self.cell_at(column)
        if cell:
            return cell.state
        return None


class State(models.Model):
    text = models.CharField(max_length=255, blank=True, null=True, help_text="A regularized form for the text of this state.")

    def __str__(self):
        cell = self.cells().first()
        if cell and cell.token:
            return str(cell.token)

        if self.text:
            return self.text
        return "OMIT"

    def str_at(self, column):
        cell = self.cells_at(column).first()
        if cell and cell.token:
            return str(cell.token)
        return str(self)
        
    def rows(self):
        return Row.objects.filter(cell__state=self)

    def cells(self):
        return Cell.objects.filter(state=self)

    def cells_at(self, column):
        return self.cells().filter(column=column)


class Column(models.Model):
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    order = models.PositiveIntegerField("The rank of this column in the alignment")
    atext = models.ForeignKey( State, on_delete=models.SET_DEFAULT, null=True, blank=True, default=None )
    atext_notes = models.TextField(default=None, null=True, blank=True)

    class Meta:
        ordering = ['alignment', 'order']

    def __str__(self):
        return f"{self.alignment}:{self.order}"

    def get_transition(self, start_state, end_state):
        transition = self.transition_set.filter(start_state=start_state, end_state=end_state).first()
        if transition:
            return transition
        inverse_transition = self.transition_set.filter(start_state=end_state, end_state=start_state).first()
        if inverse_transition:
            return inverse_transition.create_inverse()
        return None

    def max_token_len(self):
        value = 0
        for cell in self.cell_set.all():
            token_text = "-" if not cell.token else cell.token.text
            value = max(value, len(token_text))
        return value

    def only_punctuation(self):
        states = self.states()
        for state in states:
            if state.text not in [None, ""]:
                return False
        return True

    def is_empty(self):
        return self.cell_set.exclude(token=None).count() == 0

    def states(self):
        return State.objects.filter(cell__column=self).distinct()

    def state_count(self):
        return len(self.states())

    def state_pairs(self):
        import itertools
        states = list(self.states())
        return list(itertools.combinations(states, 2))

    def cells_with_state(self, state):
        return Cell.objects.filter(column=self, state=state)
    
    def rows_with_state(self, state):
        cells = self.cells_with_state(state)
        return Row.objects.filter(cell__in=cells)

    def next_pair( self, pair_rank ):
        pairs = self.state_pairs()
        
        # Check pairs on this column
        if pair_rank + 1 < len(pairs):
            return self, pair_rank + 1

        # Check pairs on this alignment
        for column in self.alignment.column_set.filter(order__gt=self.order):
            pairs = column.state_pairs()
            if len(pairs):
                return column, 0
        
        # Check next alignment
        for alignment in Alignment.objects.filter( verse__id__gt=self.alignment.verse.id ):
            for column in alignment.column_set.all():
                pairs = column.state_pairs()
                if len(pairs):
                    return column, 0
        return None,None

    def transition_for_pair( self, pair_rank ):
        pairs = self.state_pairs()
        if pair_rank >= len(pairs):
            return None
        pair = pairs[pair_rank]
        start_state = pair[0]
        end_state = pair[1]
        return Transition.objects.filter(column=self, start_state=start_state, end_state=end_state ).first()

    def set_transition(self,start_state, end_state, transition_type, inverse):
        transition, _ = Transition.objects.update_or_create( column=self, start_state=start_state, end_state=end_state, defaults={
            'transition_type': transition_type,
            'inverse': inverse,
        })
        return transition

    def next_untagged_pair( self, pair_rank ):
        column = self
        transition = True
        while column is not None and transition is not None:
            column, pair_rank = column.next_pair( pair_rank )
            if column:
                transition = column.transition_for_pair(pair_rank)
        return column, pair_rank

    def prev_pair( self, pair_rank ):
        pairs = self.state_pairs()
        
        # Check pairs on this column
        if pair_rank - 1 >= 0:
            return self, pair_rank-1

        # Check pairs on this alignment
        for column in self.alignment.column_set.filter(order__lt=self.order).reverse():
            pairs = column.state_pairs()
            if len(pairs):
                return column, len(pairs)-1
        
        # Check prev alignment
        for alignment in Alignment.objects.filter( verse__lt=self.alignment.verse ).reverse():
            for column in alignment.column_set.all().reverse():
                pairs = column.state_pairs()
                if len(pairs):
                    return column, len(pairs)-1
        return None,None        

    def pair_url( self, pair_rank ):
        if pair_rank is None:
            return ""
        return reverse( 'classify_transition_for_pair', kwargs={
            "family_siglum":self.alignment.family.name,
            "verse_ref": self.alignment.verse.url_ref(),
            "column_rank": self.order,
            "pair_rank": pair_rank,
        })

    def next_pair_url( self, pair_rank ):
        next_column, next_pair_rank = self.next_pair( pair_rank )
        if next_column is None:
            return ""
        return next_column.pair_url( next_pair_rank )

    def next_untagged_pair_url( self, pair_rank ):
        next_column, next_pair_rank = self.next_untagged_pair( pair_rank )
        if next_column is None:
            return ""
        return next_column.pair_url( next_pair_rank )

    def prev_pair_url( self, pair_rank ):
        prev_column, prev_pair_rank = self.prev_pair( pair_rank )
        if prev_column is None and prev_pair_rank is None:
            return ""
        return prev_column.pair_url( prev_pair_rank )

    

class Token(models.Model):
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    text = models.CharField(max_length=255, help_text="The characters of this token/word as they appear in the manuscript text.")
    regularized = models.CharField(max_length=255, help_text="A regularized form of the text of this token.")
    rank = models.PositiveIntegerField(blank=True, null=True, default=None) # DEPRECATED

    def __str__(self):
        return self.text


class Cell(models.Model):
    row = models.ForeignKey( Row, on_delete=models.CASCADE )
    column = models.ForeignKey( Column, on_delete=models.CASCADE )
    token = models.ForeignKey( Token, on_delete=models.CASCADE, blank=True, null=True )
    state = models.ForeignKey( State, on_delete=models.CASCADE, blank=True, null=True )

    class Meta:
        ordering = ['column', 'row']

    def token_display(self):
        if self.token:
            return str(self.token)
        return ""

class TransitionType(models.Model):
    name = models.CharField(max_length=255)
    inverse_name = models.CharField(max_length=255, blank=True, null=True, default=None)

    def __str__(self):
        if not self.inverse_name:
            return self.name
        return f"{self.name} <--> {self.inverse_name}"

    def str_with_direction(self, is_inverse):
        return self.inverse_name if is_inverse and self.inverse_name else self.name

    class Meta:
        ordering = ['name']


class TransitionClassifier(PolymorphicModel):
    name = models.CharField(max_length=255, unique=True)    
    transition_type = models.ForeignKey( TransitionType, on_delete=models.CASCADE )

    def __str__(self):
        return self.name

    def match(self, column, start_state, end_state):
        """ Returns True if this these states should be classified with this object's transition type. """
        raise NotImplementedError("This method is not implemented.")

    def classify(self, column, start_state, end_state):
        transition = None
        if self.match(column, start_state, end_state):
            transition, _ =  Transition.objects.update_or_create(
                column=column,
                start_state=start_state,
                end_state=end_state,
                defaults=dict(
                    inverse=False,
                    transition_type=self.transition_type,
                    classifier=self,
                )
            )
        elif self.match( column, end_state, start_state ):
            transition, _ = Transition.objects.update_or_create(
                column=column,
                start_state=start_state,
                end_state=end_state,
                defaults=dict(
                    inverse=True,
                    transition_type=self.transition_type,
                    classifier=self,
                )
            )
        return transition


class RegexTransitionClassifier(TransitionClassifier):
    start_state_regex = models.CharField(max_length=255)
    end_state_regex = models.CharField(max_length=255)
    

    def match(self, column, start_state, end_state):
        if re.match( str(self.start_state_regex), str(start_state.text) ) == None:
            return False
        if re.match( str(self.end_state_regex), str(end_state.text) ) == None:
            return False
        return True


class Transition(models.Model):
    column = models.ForeignKey( Column, on_delete=models.CASCADE )
    transition_type = models.ForeignKey( TransitionType, on_delete=models.CASCADE )
    inverse = models.BooleanField()
    start_state = models.ForeignKey( State, on_delete=models.CASCADE, related_name="start_state" )
    end_state = models.ForeignKey( State, on_delete=models.CASCADE, related_name="end_state" )
    classifier = models.ForeignKey( 
        TransitionClassifier, 
        null=True, 
        blank=True, 
        default=None, 
        on_delete=models.SET_DEFAULT,
        help_text="The transition classifer used to automatically assign the transition type for these states." 
    )

    def __str__(self):
        t = self.transition_type_str()
        return f"'{self.start_state}' → '{self.end_state}' ({t})"

    def transition_type_str(self):
        return self.transition_type.str_with_direction(self.inverse)

    def inverse_transition_type_str(self):
        return self.transition_type.str_with_direction(not self.inverse)

    def create_inverse(self):
        return Transition(column=self.column, transition_type=self.transition_type, inverse=(not self.inverse), start_state=self.end_state, end_state=self.start_state)




class Rate(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

    def symbol(self):
        return self.name.replace(" ","")


class RateSystem(models.Model):
    name = models.CharField(max_length=255)
    default_rate = models.ForeignKey( Rate, on_delete=models.SET_DEFAULT, default=None, blank=True, null=True )

    def __str__(self):
        return self.name

    def get_transition_rate(self, transition):
        transition_rate = self.transitionrate_set.filter(transition_type=transition.transition_type, inverse=transition.inverse).first()
        if transition_rate:
            return transition_rate
        
        transition_rate = self.transitionrate_set.filter(transition_type=transition.transition_type, inverse=None).first()
        if transition_rate:
            return transition_rate

        return self.default_rate


class TransitionRate(models.Model):
    system = models.ForeignKey( RateSystem, on_delete=models.CASCADE )
    rate = models.ForeignKey( Rate, on_delete=models.CASCADE )
    transition_type = models.ForeignKey( TransitionType, on_delete=models.CASCADE )
    inverse = models.BooleanField(default=None, blank=True, null=True, help_text="If None, then the rate applies in both directions.")

    def __str__(self):
        transition_string = self.transition_type.str_with_direction(self.inverse) if self.inverse else str(self.transition_type)
        return f"'{transition_string}' at rate '{self.rate}' in '{self.system}'"


class TransitionTypeToIgnore(models.Model):
    transition_type = models.OneToOneField( TransitionType, on_delete=models.CASCADE, help_text="The transition type to treat as non-significant for analysis." )

    def __str__(self):
        return str(self.transition_type)
