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
        self.alignment, _ = Alignment.objects.update_or_create( family=family, verse=verse, word_to_id=word_to_id, id_to_word=id_to_word )
        self.tokens = np.asarray( [-1, 5, 4, 3, 2, 1, -1] )
        self.original_columns_count = len(self.tokens)
        for row_index in range(10):
            transcription = baker.make(VerseTranscription, verse=verse )
            Row.objects.update_or_create( alignment=self.alignment, tokens=self.tokens, transcription=transcription )
        
        for order in range(len(self.tokens)):
            Column.objects.update_or_create( alignment=self.alignment, order=order)

        

    def test_empty_columns(self):
        empty = self.alignment.empty_columns()
        gold = [True] + [False]*5 + [True]
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
            self.assertEqual( len(row.tokens), col_count )
            self.assertEqual( np.any(row.tokens == -1), False )

        for index, column in enumerate(self.alignment.column_set.all()):
            self.assertEqual( index, column.order )

    def check_shift(self, delta, initial_col_order):
        column = Column.objects.get(alignment=self.alignment, order=initial_col_order)
        row = self.alignment.row_set.first()
        token_id = row.token_id_at(column)

        self.alignment.shift(row=row, column=column, delta=delta)
        self.assertEqual( 6, self.alignment.column_set.count() )

        row = self.alignment.row_set.first() # Get the row again from the database
        column = Column.objects.get(id=column.id) # Get the column again from the database

        final_column_order = initial_col_order if delta > 0 else initial_col_order + 1
        final_column_order -= 1 # Because the first column is deleted in the clean up

        self.assertEqual( column.order, final_column_order )
        self.assertEqual( row.token_id_at(column), -1 )
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
        


