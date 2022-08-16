from django.core.management.base import BaseCommand, CommandError

from dcodex.models import VerseTranscriptionBase, Manuscript
from dcodex_collation.latex import write_latex

from ._mixins import VersesCommandMixin


class Command(VersesCommandMixin, BaseCommand):
    help = "Creates a LaTeX file from an alignment."

    def add_arguments(self, parser):
        self.add_verses_parser(parser, family_optional=False, start_optional=True)
        parser.add_argument("-o", "--output", type=str, help="An output LaTeX file.")
        parser.add_argument(
            "-x",
            "--exclude",
            type=str,
            nargs="+",
            help="A list of witnesses to exclude.",
        )
        parser.add_argument(
            "-w",
            "--witnesses",
            type=str,
            nargs="+",
            help="Restrict to just witnesses in this list.",
        )
        parser.add_argument(
            "-a",
            "--atext",
            action="store_true",
            default=False,
            help="Includes the A-Text as a witness. Default False.",
        )

    def handle(self, *args, **options):
        family, verses = self.get_family_and_verses_from_options(options)
        witnesses_in_family = family.manuscripts()

        # Filter for witnesses that attest verses in this selection
        witness_ids = []
        for witness in witnesses_in_family.all():
            if (
                VerseTranscriptionBase.objects.filter(
                    verse__in=verses, manuscript=witness
                ).count()
                > 0
            ):
                if witness.siglum.endswith("_C"):
                    continue
                witness_ids.append(witness.id)
        witnesses = witnesses_in_family.filter(
            id__in=witness_ids
        )  # Should do this a better way

        # Exclude
        if options["witnesses"]:
            restrict_ids = []
            for siglum in options["witnesses"]:
                ms = Manuscript.find(siglum)
                if not ms:
                    raise Exception(f"Cannot find ms {siglum}.")
                restrict_ids.append(ms.id)

            witnesses = witnesses.filter(id__in=restrict_ids)

        # Exclude
        if options["exclude"]:
            exclude_ids = []
            for siglum in options["exclude"]:
                ms = Manuscript.find(siglum)
                if not ms:
                    raise Exception(f"Cannot find ms {siglum}.")
                exclude_ids.append(ms.id)

            witnesses = witnesses.exclude(id__in=exclude_ids)

        if options["output"]:
            with open(options["output"], "w") as file:
                write_latex(family, verses, witnesses, file, atext=options["atext"])
        else:
            write_latex(family, verses, witnesses, atext=options["atext"])
