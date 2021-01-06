# Preparing the spaCy document
from spacy import displacy

from loadSpacy1 import nlp

text = 'Tony Stark owns the company StarkEnterprises . Emily Clark works at Microsoft and lives in Manchester. She ' \
       'loves to read the Bible and learn French '
doc = nlp(text)

# Printing the named entities
print(doc.ents)

for ent in doc.ents:
    print(f"{ent} may be {ent.label_} by spacy.")

displacy.render(doc, style='ent')
