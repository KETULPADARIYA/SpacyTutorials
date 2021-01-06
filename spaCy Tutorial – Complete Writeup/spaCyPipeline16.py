# TOkenizer,Tagger, DependencyParser, EntityRecognizer,TextCategorizer,EntityRuler,Sentencizer,
# merge_noun_chunks [after tagger and parser], merge_entities [afrer ner ] , merge_subtokens
from loadSpacy1 import nlp

print(nlp.pipe_names)

# to check particular component is present in pipeline through nlp.has_pipe .

print(nlp.has_pipe("textcat"))

# How to add a component to the pipeline ?
# nlp.add_pipe(nlp.create_pipe("textcat"))
# print(nlp.pipe_names)

# Observe that textcat has been added at the last.
# The order of the components signify the order in which the Doc will be processed.

nlp.add_pipe(nlp.create_pipe("textcat"),before="ner")
print(nlp.pipe_names)


# how to remove, replace and rename pipeline components ?
print(f"Pipeline components present :- {nlp.pipe_names}")

nlp.remove_pipe("textcat")
print(f"After removing the textcat pipeline :-{nlp.pipe_names}")

