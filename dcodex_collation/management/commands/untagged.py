from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from django.db.models import F
from django.db.models import Count, Sum


class Command(BaseCommand):
    help = 'Displays columns with untagged transitions.'

    def add_arguments(self, parser):
        parser.add_argument('family', type=str, help="The siglum for a family of manuscripts.")

    def handle(self, *args, **options):
        family = Family.objects.get(name=options['family'])

        columns = (
            Column.objects.filter(alignment__family=family )
                .annotate(transition_count=Count('transition', distinct=True))
                .annotate(state_count=Count('cell__state', distinct=True))
                .filter(state_count__gt=1)
                .filter(transition_count__lt=(F("state_count")*(F("state_count")-1)/2) ) 
                .annotate(untagged_count=(F("state_count")*(F("state_count")-1)/2-F("transition_count")))
                .order_by("alignment__verse__rank", 'order')

        )
        count = columns.aggregate(Sum('untagged_count'))['untagged_count__sum']
        print(f"There are {count} unclassified transitions.")
        print("These are in the following columns:")
        for column in columns:
            # print(column, column.transition_count, column.state_count, column.untagged_count)
            print(column)            