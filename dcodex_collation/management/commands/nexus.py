from dcodex.models import * 
from dcodex_collation.nexus import write_nexus

from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = "Creates a NEXUS file from an alignment."

    def add_arguments(self, parser):
        parser.add_argument('family', type=str, help="The siglum for a family of manuscripts.")
        parser.add_argument('start', type=str, help="The starting verse of the passage selection.")
        parser.add_argument('end', type=str, nargs='?', help="The ending verse of the passage selection. If this is not given, then it only aligns the start verse.")
        parser.add_argument('--filename', type=str, help="The path to the NEXUS file to be outputted.")
        parser.add_argument('--exclude-regex', type=str, help="A regex to exclude.")


    def handle(self, *args, **options):
        family = Family.objects.get(name=options['family'])
        witnesses_in_family = family.manuscripts()

        VerseClass = witnesses_in_family.first().verse_class()

        start_verse_string = options['start'] or ""
        end_verse_string = options['end'] or ""

        verses = VerseClass.queryset_from_strings( start_verse_string, end_verse_string )

        # Filter for witnesses that attest verses in this selection
        witness_ids = []
        for witness in witnesses_in_family.all():
            if VerseTranscriptionBase.objects.filter( verse__in=verses, manuscript=witness ).count() > 0:
                if witness.siglum.endswith("_C"):
                    continue
                witness_ids.append( witness.id )
        witnesses = witnesses_in_family.filter(id__in=witness_ids) # Should do this a better way

        if options['filename']:
            with open(options['filename'], 'w') as file:
                write_nexus( family, verses, witnesses, file )
        else:
            write_nexus( family, verses, witnesses )




