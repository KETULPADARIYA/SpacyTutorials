# While trying to detect entities, some times certain names or organizations are not recognized by default. It might
# be because they are small scale or rare.
# Wouldn’t it be better to improve accuracy of our doc.ents_ method ?

# spaCy provides a more advanced component EntityRuler that let’s you match named entities based on pattern
# dictionaries. Overall, it makes Named Entity Recognition more efficient.


from spacy.pipeline import EntityRuler

from loadSpacy1 import nlp

# Initialize the EntityRuler
ruler = EntityRuler(nlp)

# Basically, you need to pass a list of dictionaries,
# where each dictionary represents a pattern to be matched.
pattern = [
    {
        "label": 'WORK_OF_ART',
        "pattern": 'My guide to statistics'
    }
]


ruler.add_patterns(pattern)

nlp.add_pipe(ruler)

doc = nlp(" I recently published my work fanfiction by Dr.X . Right now I'm studying the book of my friend ."
          "You should try My guide to statistics for clear concepts.")

print([(ent.text, ent.label) for ent in doc.ents])
