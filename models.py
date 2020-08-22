from django.db import models
from scipy.cluster import hierarchy

def tokenize_strings( transcriptions ):
    return [transcription.transcription.split() for transcription in transcriptions]

def align_family_at_verse(family, verse):
    transcriptions = list(family.transcriptions_at(location.start_verse))

    # Distance matrix
    distance_matrix_as_vector = []
    for x_index, x in enumerate(transcriptions):
        for y_index, y in enumerate(transcriptions):
            if y_index >= x_index:
                break

            distance = gotoh_counts.distance( x, y, *gotoh_param )
            distance_matrix_as_vector.append( distance )

    # Guide tree
    method="average"
    linkage = hierarchy.linkage(distance_matrix_as_vector, method)

    # Tokenise
    token_strings = tokenize_strings(transcriptions)
    vocab = {}
    vocab_index = 0
    alignments = []
    for transcription_token_strings in token_strings:
        transcription_tokens = []
        for token_string in transcription_tokens:
            if token_string not in vocab:
                vocab[token_string] = vocab_index
                vocab_index += 1
            transcription_tokens.append(vocab[token_string])
        alignments.append( np.expand_dims( np.asarray(transcription_tokens ), axis=1) )


    # Alignment
    for link_index, link_row in enumerate(linkage):
        node_id = len(transcriptions) + link_index
        left_node_id = int(link_row[0])
        right_node_id = int(link_row[1])

        left_alignment = alignments[left_node_id]
        right_alignment = alignments[right_node_id]

        new_alignment, flip = gotoh_msa( left_alignment, right_alignment, gap_open=-10, gap_extend=-1 )
        alignments.append( new_alignment )

    print(alignments)
    
    # Save