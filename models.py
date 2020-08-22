from django.db import models
from scipy.cluster import hierarchy
import gotoh_counts
import gotoh_msa
import numpy as np
from jsonfield import JSONField
from ndarray import NDArrayField

from dcodex.models import *

def tokenize_strings( transcriptions ):
    return [transcription.transcription.split() for transcription in transcriptions]

def align_family_at_verse(family, verse, gotoh_param):
    transcriptions = list(family.transcriptions_at(verse))

    # Distance matrix
    distance_matrix_as_vector = []
    for x_index, x in enumerate(transcriptions):
        for y_index, y in enumerate(transcriptions):
            if y_index >= x_index:
                break

            distance = gotoh_counts.weighted_nonmatches( x.transcription, y.transcription, *gotoh_param )
            distance_matrix_as_vector.append( distance )

    # Guide tree
    method="average"
    linkage = hierarchy.linkage(distance_matrix_as_vector, method)

    # Tokenise
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


    # Alignment
    alignment_transcriptions = [[transcription] for transcription in transcriptions]
    for link_index, link_row in enumerate(linkage):
        node_id = len(transcriptions) + link_index
        left_node_id = int(link_row[0])
        right_node_id = int(link_row[1])

        left_alignment = alignments[left_node_id]
        right_alignment = alignments[right_node_id]

        new_alignment = gotoh_msa.align( left_alignment, right_alignment, gap_open=-10, gap_extend=-1 )
        alignments.append( new_alignment )
        alignment_transcriptions.append(  alignment_transcriptions[left_node_id] + alignment_transcriptions[right_node_id] )

    alignment_array = alignments[-1]
    print("alignment", alignment.shape)
    print("len(alignment_transcriptions[-1])", len(alignment_transcriptions[-1]))

    print(alignment_transcriptions[-1])


    alignment, _ = Alignment.objects.update_or_create( family=family, verse=verse, defaults={
        "word_to_id": vocab,
        "id_to_word": np.asarray( list(vocab.keys()) ),
        "matrix_data": np.asarray( list(vocab.keys()) ),
    } )
    #alignment = Alignment( family=family, verse=verse )
    #alignment.save()

    #alignment.word_to_id = vocab
    #alignment.id_to_word = np.asarray( list(vocab.keys()) )

    id_to_word = np.asarray( list(vocab.keys()) )
    print("id_to_word", id_to_word)
    #return

    for transcription, tokens in zip( alignment_transcriptions[-1], np.rollaxis(alignment_array, 1) ):
        aligned_transcription, _ = AlignedTranscription.objects.update_or_create( transcription=transcription, alignment=alignment, defaults={
            "tokens": tokens,
        } )

        # TODO Save Columns
        
        print("transcription", transcription)
        print("tokens", tokens)
        for token in tokens:
            if token == -1:
                print("GAP")
            else:
                print(token, id_to_word[token])


class Alignment(models.Model):
    family = models.ForeignKey( Family, on_delete=models.SET_DEFAULT, default=None, null=True, blank=True )
    verse = models.ForeignKey( Verse, on_delete=models.CASCADE )
    word_to_id = JSONField(help_text="Vocab dictionary")
    id_to_word = NDArrayField(help_text="Index of vocab dictionary")
    matrix_data = NDArrayField(help_text="Numpy array for the similarity matrix")


class AlignedTranscription(models.Model):
    transcription = models.ForeignKey( VerseTranscription, on_delete=models.CASCADE )
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    tokens = NDArrayField(help_text="Numpy array for the tokens. IDs correspond to the vocab in the alignment")


class Column(models.Model):
    alignment = models.ForeignKey( Alignment, on_delete=models.CASCADE )
    order = models.PositiveIntegerField("The rank of this column in the alignment")

    class Meta:
        ordering = ['order']


        
