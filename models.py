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

def tokenize_string( string ):
    string = string.replace("."," .")
    string = re.sub("\s+"," ", string)
    string = remove_markup(string)
    return string.split()

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
    


    alignment, _ = Alignment.objects.update_or_create( family=family, verse=verse, defaults={
        "word_to_id": vocab,
        "id_to_word": id_to_word,
    } )
    #alignment = Alignment( family=family, verse=verse )
    #alignment.save()

    #alignment.word_to_id = vocab
    #alignment.id_to_word = np.asarray( list(vocab.keys()) )

    id_to_word = np.asarray( list(vocab.keys()) )
    #print("id_to_word", id_to_word)
    #return

    
    for order in range( alignment_array.shape[0] ):
        column, _ = Column.objects.update_or_create( order=order, alignment=alignment, defaults={} )
    Column.objects.filter(alignment=alignment, order__gte=alignment_array.shape[0] ).delete()

    Row.objects.filter(alignment=alignment).delete()
    for transcription, tokens in zip( alignment_transcriptions, np.rollaxis(alignment_array, 1) ):
        aligned_transcription, _ = Row.objects.update_or_create( transcription=transcription, alignment=alignment, defaults={
            "tokens": tokens,
        } )

    #dn = hierarchy.dendrogram(linkage, orientation='right',labels=[transcription.manuscript.siglum for transcription in transcriptions])
    #plt.show()
    return alignment


class Alignment(models.Model):
    family = models.ForeignKey( Family, on_delete=models.SET_DEFAULT, default=None, null=True, blank=True )
    verse = models.ForeignKey( Verse, on_delete=models.CASCADE )
    word_to_id = JSONField(help_text="Vocab dictionary")
    id_to_word = NDArrayField(help_text="Index of vocab dictionary")


class Row(models.Model):
    transcription = models.ForeignKey( VerseTranscription, on_delete=models.CASCADE )
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    tokens = NDArrayField(help_text="Numpy array for the tokens. IDs correspond to the vocab in the alignment")


class Column(models.Model):
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    order = models.PositiveIntegerField("The rank of this column in the alignment")

    class Meta:
        ordering = ['order']


        
