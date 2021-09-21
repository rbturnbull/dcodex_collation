from django.core.management.base import BaseCommand, CommandError
import pandas as pd
from dcodex.models import *
from dcodex_collation.models import Alignment


class Command(BaseCommand):
    help = 'Deletes transitions where the columns no longer have those states.'

    def handle(self, *args, **options):

        for alignment in Alignment.objects.all():
            alignment.delete_invalid_transitions()
