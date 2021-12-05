from django.core.management.base import BaseCommand, CommandError

from dcodex.models import *
from dcodex_collation.models import *

from ._mixins import VersesCommandMixin

class Command(VersesCommandMixin, BaseCommand):
    help = "Classifies transitions with a list of TransitionClassifier objects."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)
        parser.add_argument(
            "classifiers",
            type=str,
            nargs="?",
            help="The classifiers to use. If none specified, then all are used from the database.",
        )

    def handle(self, *args, **options):

        classifiers = TransitionClassifier.objects.all()
        if options["classifiers"]:
            classifiers = classifiers.filter(name__in=[options["classifiers"]])

        if classifiers.count() == 0:
            print(f"No classifiers found. Command-Line Arguments: {options['classifiers']}")
            return

        columns = self.get_columns_from_options(options)
        for column in columns:
            for pair_rank, pair in enumerate(column.state_pairs()):

                if not column.transition_for_pair(pair_rank):
                    start_state = pair[0]
                    end_state = pair[1]
                    print(column, pair_rank, start_state, end_state)

                    for classifier in classifiers:
                        transition = classifier.classify(
                            column, start_state, end_state
                        )

                        if transition:
                            print(f"classifier {classifier} matched.")
                            break
