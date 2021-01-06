import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_sm")

# must be same length of words and spaces or not provied  spaces


# spaces is a list of boolean values indicating if subsequent tokens
# are followed by any whitespace
# so, create a Spacy document with your tokenisation
words = ['nuts', 'itch', "india"]
spaces = [True, True,False]

doc = spacy.tokens.doc.Doc(
    nlp.vocab, words=words)
# run the standard pipeline against it
for name, proc in nlp.pipeline:
    doc = proc(doc)
    print(doc,name)

print(doc.ents)