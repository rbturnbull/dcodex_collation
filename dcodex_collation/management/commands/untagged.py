from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from django.db.models import F
from django.db.models import Count, Sum
from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Displays columns with untagged transitions."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)

    def handle(self, *args, **options):
        columns = self.get_columns_from_options(options)

        columns = (
            columns
            .annotate(transition_count=Count("transition", distinct=True))
            .annotate(state_count=Count("cell__state", distinct=True))
            .filter(state_count__gt=1)
            .filter(
                transition_count__lt=(F("state_count") * (F("state_count") - 1) / 2)
            )
            .annotate(
                untagged_count=(
                    F("state_count") * (F("state_count") - 1) / 2
                    - F("transition_count")
                )
            )
            .order_by("alignment__verse__rank", "order")
        )
        count = columns.aggregate(Sum("untagged_count"))["untagged_count__sum"]
        print(f"There are {count} unclassified transitions.")
        print("These are in the following columns:")
        for column in columns:
            # print(column, column.transition_count, column.state_count, column.untagged_count)
            print(column)
