import sys
from .models import *
from lxml import etree

def write_nexus(
    family, 
    verses, 
    witnesses=None, 
    file=None, 
    allow_ignore=True, 
    atext=False,
    charstatelabels=True,
):
    file = file or sys.stdout
    witnesses = witnesses or family.manuscripts()


    root = etree.Element("TEI")
    print(root)


# <?xml version='1.0' encoding='UTF-8'?>
# <TEI xmlns="http://www.tei-c.org/ns/1.0">
#     <teiHeader>
#         <fileDesc>
#             <titleStmt>
#                 <title>A limited collation of Ephesians at selected variation units from the United Bible Societies Greek New Testament, 5th Edition</title>
#             </titleStmt>
#             <publicationStmt>
#                 <authority xml:id="JJM">Edited by Joey McCollum</authority>
#                 <date>2022</date>
#                 <availability>
#                     <p>This is an open access work licensed under a Creative Commons Attribution 4.0 International license.</p>
#                 </availability>
#             </publicationStmt>
