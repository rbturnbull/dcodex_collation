from dcodex.models import *
from dcodex_collation.nexus import write_nexus

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Creates a NEXUS file from an alignment."

    def add_arguments(self, parser):
        parser.add_argument(
            "family", type=str, help="The siglum for a family of manuscripts."
        )
        parser.add_argument(
            "start", type=str, help="The starting verse of the passage selection."
        )
        parser.add_argument(
            "end",
            type=str,
            nargs="?",
            help="The ending verse of the passage selection. If this is not given, then it only aligns the start verse.",
        )
        parser.add_argument(
            "-f", "--file", type=str, help="The path to the NEXUS file to be outputted."
        )
        parser.add_argument(
            "-x",
            "--exclude",
            type=str,
            nargs="+",
            help="A list of witnesses to exclude.",
        )
        parser.add_argument(
            "-k", "--skip", type=str, nargs="+", help="A list of verses to skip."
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
        family = Family.objects.get(name=options["family"])
        witnesses_in_family = family.manuscripts()

        VerseClass = witnesses_in_family.first().verse_class()

        start_verse_string = options["start"] or ""
        end_verse_string = options["end"] or ""

        verses = VerseClass.queryset_from_strings(start_verse_string, end_verse_string)

        if options["skip"]:
            verse_ids_to_skip = [
                VerseClass.get_from_string(verse_ref_to_skip).id
                for verse_ref_to_skip in options["skip"]
            ]
            verses = verses.exclude(id__in=verse_ids_to_skip)

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

        if options["file"]:
            with open(options["file"], "w") as file:
                write_nexus(family, verses, witnesses, file, atext=options["atext"])
        else:
            write_nexus(family, verses, witnesses, atext=options["atext"])
