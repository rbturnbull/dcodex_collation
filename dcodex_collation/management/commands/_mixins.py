from dcodex.models import Family
from dcodex_collation.models import Alignment, Column

class VersesCommandMixin():

    def add_verses_parser(self, parser, family_optional=False, start_optional=False):
        family_args = ['-f', '--family'] if family_optional else ['family']
        parser.add_argument(*family_args, type=str, default=None, help="The family to use in this command.")
        start_args = ['-s', '--start'] if start_optional else ['start']
        parser.add_argument(
            *start_args,
            default="",
            type=str, help="The starting verse of the passage selection."
        )
        parser.add_argument(
            '-e',
            "--end",
            type=str,
            help="The ending verse of the passage selection. If this is not given, then it only uses the start verse.",
        )
        parser.add_argument(
            '-k',
            "--skip", 
            type=str, nargs="+", help="A list of verses to skip."
        )

    def get_family_from_options(self, options):
        if ('family' in options) and options["family"]:
            return Family.objects.get(name=options["family"])
        return None        

    def get_verses_from_options(self, options, verse_class=None):
        if verse_class is None:
            family = self.get_family_from_options(options)
            if family is None:
                raise ValueError("verse_class is none and it is needed if a family is not present.")
            witnesses_in_family = family.manuscripts()
            verse_class = witnesses_in_family.first().verse_class()

        start_verse_string = options["start"] or ""
        end_verse_string = options["end"] or ""

        verses = verse_class.queryset_from_strings(start_verse_string, end_verse_string)

        if options["skip"]:
            verse_ids_to_skip = set(
                [
                    verse_class.get_from_string(verse_ref_to_skip).id
                    for verse_ref_to_skip in options["skip"]
                ]
            )
            verses = verses.exclude(id__in=verse_ids_to_skip)

        return verses

    def get_family_and_verses_from_options(self, options, verse_class=None):    
        family = self.get_family_from_options(options)    
        verses = self.get_verses_from_options(options, verse_class=verse_class)
        return family, verses

    def get_alignments_from_options(self, options, verse_class=None):
        family, verses = self.get_family_and_verses_from_options(options, verse_class=verse_class)    

        alignments = Alignment.objects.filter(verse__in=verses)
        if family:
            alignments = alignments.filter(family=family)

        return alignments

    def get_columns_from_options(self, options, verse_class=None):
        alignments = self.get_alignments_from_options(options, verse_class=verse_class)
        return Column.objects.filter(alignment__in=alignments)