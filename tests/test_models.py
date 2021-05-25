from django.test import TestCase

from model_bakery import baker

from dcodex.models import *
from dcodex_collation.models import *
import numpy as np

class AlignmentFromTokensTest(TestCase):
    def setUp(self):
        family = baker.make(Family)
        verse = baker.make(Verse)
        word_to_id = {str(x):x for x in range(10)}
        id_to_word = np.asarray( list(word_to_id.keys() ) )
        self.alignment, _ = Alignment.objects.update_or_create( family=family, verse=verse )
        self.tokens = np.asarray( [-1, 5, 4, 3, 2, 1, -1] )
        self.original_columns_count = len(self.tokens)
        for row_index in range(10):
            transcription = baker.make(VerseTranscription, verse=verse )
            row, _ = Row.objects.update_or_create( alignment=self.alignment, transcription=transcription )
            row.create_cells_for_tokens(self.tokens, id_to_word)

    def test_empty_columns(self):
        empty = self.alignment.empty_columns()
        gold = [True] + [False]*5 + [True]
        # print(f'{empty =}')

        # for row in self.alignment.row_set.all():
        #     print('row', row)
        #     for cell in row.cell_set.all():
        #         print(cell, 'cell')
        np.testing.assert_array_equal( empty, np.asarray(gold))
        self.assertEqual( np.argwhere(empty)[0][0], 0 )
        self.assertEqual( np.argwhere(empty)[1][0], 6 )

    def test_add_column(self):
        column_index = 3
        self.alignment.add_column( column_index )
        empty = self.alignment.empty_columns()

        self.assertEqual( empty[column_index], True )
        self.assertEqual( self.alignment.column_set.count(), self.original_columns_count + 1 )
        for index, column in enumerate(self.alignment.column_set.all()):
            self.assertEqual( index, column.order )

    def test_clear_empty(self):
        start_empty = self.alignment.empty_columns()
        self.alignment.clear_empty()
        empty = self.alignment.empty_columns()

        self.assertEqual( np.any(empty), False )
        col_count = self.alignment.column_set.count()
        self.assertEqual( col_count, 5 )
        for row in self.alignment.row_set.all():
            self.assertEqual( row.cell_set.count(), col_count ) # not sure what this is testing
            for cell in row.cell_set.all():
                self.assertEqual( cell.token is not None, True )

        for index, column in enumerate(self.alignment.column_set.all()):
            self.assertEqual( index, column.order )

    def check_shift(self, delta, initial_col_order):
        column = Column.objects.get(alignment=self.alignment, order=initial_col_order)
        row = self.alignment.row_set.first()
        token_id = row.token_id_at(column)

        self.alignment.shift(row=row, column=column, delta=delta)
        self.alignment.clear_empty()
        self.assertEqual( 6, self.alignment.column_set.count() )

        row = self.alignment.row_set.first() # Get the row again from the database
        column = Column.objects.get(id=column.id) # Get the column again from the database

        final_column_order = initial_col_order if delta > 0 else initial_col_order + 1
        final_column_order -= 1 # Because the first column is deleted in the clean up

        self.assertEqual( column.order, final_column_order )
        self.assertEqual( row.token_id_at(column), None )
        new_column = Column.objects.get(order=final_column_order+delta, alignment=self.alignment)
        self.assertEqual( row.token_id_at(new_column), token_id )

    def test_shift_positive(self):
        delta = 1
        initial_col_order = 2
        self.check_shift( delta, initial_col_order )
        
    def test_shift_negative(self):
        delta = -1
        initial_col_order = 2
        self.check_shift( delta, initial_col_order )
        


class AlignmentFromTranscriptionsTest(TestCase):
    def setUp(self):
        self.family = baker.make(Family)
        self.verse = baker.make(Verse)

        self.manuscript_sigla = [f"MS{x}" for x in range(5)]
        self.transcription_texts = [
            "A B C D E",
            "A C D E",
            "A B C D E F",
            "A B C F",
            "A B C",
        ]
        self.manuscripts = [Manuscript.objects.create(siglum=siglum, name=siglum) for siglum in self.manuscript_sigla]
        for manuscript, transcription_text in zip(self.manuscripts, self.transcription_texts):
            self.family.add_manuscript_all(manuscript)
            manuscript.save_transcription( self.verse, transcription_text )

        self.gotoh_param = [6.6995597099885345, -0.9209875054657459, -5.097397327423096, -1.3005714416503906]

    def test_align_family_at_verse(self):
        alignment = align_family_at_verse(self.family, self.verse, self.gotoh_param )

        self.assertEqual( len(self.manuscripts), alignment.row_set.count() )
        self.assertEqual( 6, alignment.column_set.count() )

    def test_ascii(self):
        alignment = align_family_at_verse(self.family, self.verse, self.gotoh_param )

        ascii = alignment.ascii()

        gold_ascii = (
            "MS0 | A | B | C | D | E | –\n" +
            "MS1 | A | – | C | D | E | –\n" +
            "MS2 | A | B | C | D | E | F\n" +
            "MS3 | A | B | C | – | – | F\n" +
            "MS4 | A | B | C | – | – | –\n" 
        )

        self.assertEqual( ascii, gold_ascii )

    def test_update_transcription_in_alignment_longer(self):
        # First create alignment
        alignment = align_family_at_verse(self.family, self.verse, self.gotoh_param )

        # Change transcription
        manuscript = self.manuscripts[-1]
        transcription = manuscript.save_transcription( self.verse, "A B C E F G H" )

        # Update alignment
        new_alignment = update_transcription_in_alignment( transcription, self.gotoh_param, gap_open=-2, gap_extend=-1 )

        self.assertEqual( alignment.id, new_alignment.id )
        self.assertEqual( len(self.manuscripts), new_alignment.row_set.count() )
        self.assertEqual( 8, new_alignment.column_set.count() )

        for index, column in enumerate(new_alignment.column_set.all()):
            # Assert column order values is sequential
            print(column, column.order)
            self.assertEqual( index, column.order )

        gold_ascii = (
            "MS0 | A | B | C | D | E | – | – | –\n" + 
            "MS1 | A | – | C | D | E | – | – | –\n" + 
            "MS2 | A | B | C | D | E | F | – | –\n" + 
            "MS3 | A | B | C | – | – | F | – | –\n" + 
            "MS4 | A | B | C | – | E | F | G | H\n"            
        )
        print(gold_ascii)
        print(new_alignment.ascii())
        self.assertEqual( new_alignment.ascii(), gold_ascii )

    def test_update_transcription_in_alignment_shorter(self):
        # First create alignment
        alignment = align_family_at_verse(self.family, self.verse, self.gotoh_param )

        # Change transcription
        manuscript = self.manuscripts[-1]
        transcription = manuscript.save_transcription( self.verse, "A G B C" )

        # Update alignment
        new_alignment = update_transcription_in_alignment( transcription, self.gotoh_param, gap_open=-2, gap_extend=-1 )

        self.assertEqual( alignment.id, new_alignment.id )
        self.assertEqual( len(self.manuscripts), new_alignment.row_set.count() )
        self.assertEqual( 7, new_alignment.column_set.count() )

        for index, column in enumerate(new_alignment.column_set.all()):
            # Assert column order values is sequential
            print(column, column.order)
            self.assertEqual( index, column.order )

        gold_ascii = (
            "MS0 | A | – | B | C | D | E | –\n" + 
            "MS1 | A | – | – | C | D | E | –\n" + 
            "MS2 | A | – | B | C | D | E | F\n" + 
            "MS3 | A | – | B | C | – | – | F\n" + 
            "MS4 | A | G | B | C | – | – | –\n"            
        )
        print(gold_ascii)
        print(new_alignment.ascii())
        self.assertEqual( new_alignment.ascii(), gold_ascii )

    def test_ignore_transitions(self):
        pass


class RegexTransitionClassifierTest(TestCase):
    def setUp(self):
        self.transition_type = TransitionType.objects.create(name="transition type 1")
        self.classifier = RegexTransitionClassifier.objects.create(
            name="classifier 1", 
            transition_type=self.transition_type,
            start_state_regex="^hello$",
            end_state_regex="^world$",
        )
        self.state1 = State.objects.create(text="hello")
        self.state2 = State.objects.create(text="world")

        self.verse = Verse.objects.create(rank=0)
        self.alignment = Alignment.objects.create(verse=self.verse)
        self.column = Column.objects.create(alignment=self.alignment, order=0)

    def test_regex_match(self):
        self.assertEqual( self.classifier.match(self.column, self.state1, self.state2), True )
        self.assertEqual( self.classifier.match(self.column, self.state2, self.state1), False )

    def test_classify(self):
        transition = self.classifier.classify(self.column, self.state1, self.state2)
        self.assertIsNotNone( transition )                        
        self.assertEqual( transition.classifier.id, self.classifier.id )        
        self.assertEqual( transition.transition_type.id, self.transition_type.id )        
        self.assertEqual( transition.inverse, False )        
        self.assertEqual( transition.start_state.id, self.state1.id )        
        self.assertEqual( transition.end_state.id, self.state2.id )        
        self.assertEqual( transition.column.id, self.column.id )        

    def test_classify_inverse(self):
        transition = self.classifier.classify(self.column, self.state2, self.state1)
        self.assertIsNotNone( transition )                        
        self.assertEqual( transition.classifier.id, self.classifier.id )        
        self.assertEqual( transition.transition_type.id, self.transition_type.id )        
        self.assertEqual( transition.inverse, True )        
        self.assertEqual( transition.start_state.id, self.state2.id )        
        self.assertEqual( transition.end_state.id, self.state1.id )        
        self.assertEqual( transition.column.id, self.column.id )        

    def test_classify_fail(self):
        transition = self.classifier.classify(self.column, self.state1, self.state1)
        self.assertIsNone( transition )       

        transition = self.classifier.classify(self.column, self.state2, self.state2)
        self.assertIsNone( transition )                        