from django.db import models
from scipy.cluster import hierarchy
import gotoh_counts
import gotoh_msa
import numpy as np
from jsonfield import JSONField
from ndarray import NDArrayField
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from dcodex.models import *
from dcodex.strings import *
from django.urls import reverse

def tokenize_string( string ):
    string = string.replace("."," .")
    string = re.sub("\s+"," ", string)
    string = remove_markup(string)
    return string.split()

GAP = -1

def tokenize_strings( transcriptions ):
    return [tokenize_string(transcription.transcription) for transcription in transcriptions]

def align_family_at_verse(family, verse, gotoh_param, iterations_count = 1, gap_open=-5, gap_extend=-2):
    transcriptions = list(family.transcriptions_at(verse))

    #transcriptions = [t for t in transcriptions if t.manuscript.siglum in ["J67_esk", "S128_esk", "CSA"]]

    # Distance matrix
    distance_matrix_as_vector = []
    for x_index, x in enumerate(transcriptions):
        x_string = normalize_transcription(x.transcription)
        for y_index, y in enumerate(transcriptions):
            if y_index >= x_index:
                break
            y_string = normalize_transcription(y.transcription)
            distance = gotoh_counts.nonmatches( x_string, y_string, *gotoh_param )
            distance_matrix_as_vector.append( distance )

    # Guide tree
    method="average"
    linkage = hierarchy.linkage(distance_matrix_as_vector, method)

    # Tokenize
    token_strings = tokenize_strings(transcriptions)
    #print(token_strings)
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
            scoring_matrix[index_i,index_j] = gotoh_counts.score( token_i, token_j, *gotoh_param )
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
        #print("--")
        #print('left sigla', [t.manuscript.siglum for t in alignment_transcriptions[left_node_id]])
        #print('right sigla', [t.manuscript.siglum for t in alignment_transcriptions[right_node_id]])
#        return 

        new_alignment = gotoh_msa.align( left_alignment, right_alignment, matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend )
        alignments.append( new_alignment )
        alignment_transcriptions.append(  alignment_transcriptions[left_node_id] + alignment_transcriptions[right_node_id] )
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

            alignment_array = gotoh_msa.align( alignment_array, row, matrix=scoring_matrix, gap_open=gap_open, gap_extend=gap_extend, visualize=False )
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
    
    for order in range( alignment_array.shape[0] ):
        column, _ = Column.objects.update_or_create( order=order, alignment=alignment, defaults={} )
    Column.objects.filter(alignment=alignment, order__gte=alignment_array.shape[0] ).delete()

    Row.objects.filter(alignment=alignment).delete()
    for transcription, tokens in zip( alignment_transcriptions, np.rollaxis(alignment_array, 1) ):
        row, _ = Row.objects.update_or_create( transcription=transcription, alignment=alignment )
        for rank, token_id in enumerate(tokens):
            column = Column.objects.get(alignment=alignment, order=rank)

            print(column)
            if token_id == -1:
                token = None
            else:
                token_text = id_to_word[token_id]
                token = Token.objects.get(text=token_text, alignment=alignment)

            # Create State
            if token and "⧙" in token.text:
                state = None
            else:
                text = token.regularized if token else None
                state, _ = State.objects.update_or_create( column=column, text=text )

            # Create Cell
            cell, _ = Cell.objects.update_or_create( row=row, column=column, defaults={
                'token':token,
                "state":state,
            })        
            #print(cell, row.transcription, column.order, token, state)

    #dn = hierarchy.dendrogram(linkage, orientation='right',labels=[transcription.manuscript.siglum for transcription in transcriptions])
    #plt.show()
    return alignment


    

class Alignment(models.Model):
    family = models.ForeignKey( Family, on_delete=models.SET_DEFAULT, default=None, null=True, blank=True )
    verse = models.ForeignKey( Verse, on_delete=models.CASCADE )
    word_to_id = JSONField(help_text="Vocab dictionary", blank=True, null=True)
    id_to_word = NDArrayField(help_text="Index of vocab dictionary", blank=True, null=True)

    def get_absolute_url(self):
        return reverse("alignment_for_family", kwargs={"family_siglum": self.family.name, "verse_ref": self.verse.url_ref() })

    def add_column(self, new_column_order):
        columns = self.column_set.filter( order__gte=new_column_order )
        for c in columns:
            c.order += 1
            c.save()
        column = Column(alignment=self, order=new_column_order)
        column.save()

        gap_state = column.gap_state()
        for row in self.row_set.all():
            Cell( row=row, column=column, state=gap_state, token=None).save()

        return column

    def empty_columns(self):
        return np.asarray( [column.is_empty() for column in self.column_set.all()] )

    def clear_empty(self):
        for column in self.column_set.all():
            if column.is_empty():
                #import logging
                #logging.warning("column empty"+ str(column.order))
                column.delete()

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
        gap_state = target_cell.state
        target_cell.token = start_cell.token
        target_cell.state = start_cell.state
        target_cell.save()

        start_cell.state = gap_state
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
        
        gap_state = target_cell.state
        start_cell = row.cell_at(column)
        target_cell.token = start_cell.token
        target_cell.state = start_cell.state
        target_cell.save()

        start_cell.state = gap_state
        start_cell.token = None
        start_cell.save()
        
        # Check that no columns are empty
        self.clear_empty( )


class Row(models.Model):
    transcription = models.ForeignKey( VerseTranscription, on_delete=models.CASCADE )
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    tokens = NDArrayField(help_text="Numpy array for the tokens. IDs correspond to the vocab in the alignment", blank=True, null=True)

    def token_id_at( self, column ):
        self.cell_at(column)
        if cell:
            return cell.token

    def token_at( self, column ):
        token_id = self.token_id_at( column )
        if token_id < 0:
            return ""
        return self.alignment.id_to_word[ token_id ]        

    def cell_at(self, column):
        return self.cell_set.filter(column=column).first()

    def text_at(self, column):
        cell = self.cell_at(column)
        if cell and cell.token and cell.token.text:
            return cell.token.text
        return ""

class Column(models.Model):
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    order = models.PositiveIntegerField("The rank of this column in the alignment")

    class Meta:
        ordering = ['order']

    def is_empty(self):
        return self.cell_set.exclude(token=None).count() == 0

    def gap_state(self):
        state, _ = State.objects.get_or_create(column=self)
        return state

    def states(self):
        if self.hasattr( 'states' ):
            return self.states

        self.states = []
        self.row_to_state = {}
        for row in self.alignment.row_set.all():
            token = row.token_at(self.order)
            if '⧙' in token:
                self.row_to_state[row.id] = None
                continue

            regularized_token = normalize_transcription(token)
            if regularized_token not in self.states:
                self.states.append( regularized_token )
            
            self.row_to_state[row.id] = self.states.index(regularized_token)
            
        return self.states

    def states_count(self):
        return len(self.states())

    def state_pairs(self):
        import itertools
        states = self.states()
        return list(itertools.combinations(states, 2))

    def rows_with_state(self, state):
        self.states()
        row_ids = [row_id for row_id in self.row_to_state if self.row_to_state[row_id] == state]
        return Row.objects.filter(id__in=row_ids)

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
        for alignment in Alignment.objects.filter( verse__gt=self.alignment.verse ):
            for column in self.alignment.column_set.all():
                pairs = column.state_pairs()
                if len(pairs):
                    return column, 0
        return None,None

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
            for column in self.alignment.column_set.all().reverse():
                pairs = column.state_pairs()
                if len(pairs):
                    return column, len(pairs)-1
        return None,None        

    def next_pair_url( self, pair_rank ):
        next_column, next_pair_rank = self.next_pair( pair_rank )
        if next_column is None and next_pair_rank is None:
            return ""
        return reverse( 'classify_transition_for_pair', kwargs={
            "family_siglum":self.alignment.family.name,
            "verse_ref": self.alignment.verse.url_ref(),
            "column_rank": next_column.order,
            "pair_rank": next_pair_rank,
        })

    def prev_pair_url( self, pair_rank ):
        prev_column, prev_pair_rank = self.prev_pair( pair_rank )
        if prev_column is None and prev_pair_rank is None:
            return ""
        return reverse( 'classify_transition_for_pair', kwargs={
            "family_siglum":self.alignment.family.name,
            "verse_ref": self.alignment.verse.url_ref(),
            "column_rank": prev_column.order,
            "pair_rank": prev_pair_rank,
        })

class State(models.Model):
    text = models.CharField(max_length=255, blank=True, null=True, help_text="A regularized form for the text of this state.")
    column = models.ForeignKey( Column, on_delete=models.CASCADE )

    def __str__(self):
        if self.text:
            return self.text
        return "OMIT"


class Token(models.Model):
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    text = models.CharField(max_length=255, help_text="The characters of this token/word as they appear in the manuscript text.")
    regularized = models.CharField(max_length=255, help_text="A regularized form of the text of this token.")
    rank = models.PositiveIntegerField()

class Cell(models.Model):
    row = models.ForeignKey( Row, on_delete=models.CASCADE )
    column = models.ForeignKey( Column, on_delete=models.CASCADE )
    token = models.ForeignKey( Token, on_delete=models.CASCADE, blank=True, null=True )
    state = models.ForeignKey( State, on_delete=models.CASCADE, blank=True, null=True )

    
class TransitionType(models.Model):
    name = models.CharField(max_length=255)
    inverse_name = models.CharField(max_length=255, blank=True, null=True, default=None)

    def __str__(self):
        if not self.inverse_name:
            return self.name
        return f"{self.name} <--> {self.inverse_name}"

    class Meta:
        ordering = ['name']


class Transition(models.Model):
    column = models.ForeignKey( Column, on_delete=models.CASCADE )
    transition_type = models.ForeignKey( TransitionType, on_delete=models.CASCADE )
    inverse = models.BooleanField()
    start_state = models.CharField(max_length=255)
    end_state = models.CharField(max_length=255)


class AText(models.Model):
    column = models.ForeignKey( Column, on_delete=models.CASCADE )
    state = models.CharField(max_length=255)

    