from django.test import TestCase

from model_bakery import baker

from dcodex.models import *
from dcodex_collation.models import *
import numpy as np

class AlignmentTest(TestCase):
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