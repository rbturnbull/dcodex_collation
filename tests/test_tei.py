from django.test import TestCase

from model_bakery import baker

from dcodex.models import Manuscript, Family, Verse
from dcodex_collation.models import Alignment, align_family_at_verse
import numpy as np


class TEITest(TestCase):
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
        self.manuscripts = [
            Manuscript.objects.create(siglum=siglum, name=siglum)
            for siglum in self.manuscript_sigla
        ]
        for manuscript, transcription_text in zip(
            self.manuscripts, self.transcription_texts
        ):
            self.family.add_manuscript_all(manuscript)
            manuscript.save_transcription(self.verse, transcription_text)

        self.gotoh_param = [
            6.6995597099885345,
            -0.9209875054657459,
            -5.097397327423096,
            -1.3005714416503906,
        ]
    
    def test_tei(self):
        alignment = align_family_at_verse(self.family, self.verse, self.gotoh_param)

        print(alignment)

