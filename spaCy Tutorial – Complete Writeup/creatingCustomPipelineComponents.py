# The parameters of add_pipe you have to provide :
#
#   >  component : You have to pass the function_name as input . This serves as our component
#
# > name : You can assign a name to the component. The component can be called using this name. If you don’t provide
#   any ,the function_name will be taken as name of the component
#
# > first,last : If you want the new component to be added first or last ,you can setfirst=True or last=True
#   accordingly.
#
# > before , after : If you want to add the component specifically before or after another component , you can use
#   these arguments.
# Note that you can set only one among first, last, before, after arguments,
# otherwise it will lead to error.
import spacy
from spacy.matcher.phrasematcher import PhraseMatcher


def my_custom_component(doc):
    doc_length = len(doc)
    print(f'The no of tokens in the document is {doc_length}')
    named_entity = [ent.label_ for ent in doc.ents]
    print(named_entity)
    return doc


nlp = spacy.load("en_core_web_sm")

nlp.add_pipe(my_custom_component, after='ner')
print(nlp.pipe_names)

doc = nlp(" The Hindu Newspaper has increased the cost. "
          "I usually read the paper on my way to Delhi railway station ")

# PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# List of book names to be matched
book_names = [
    'Pride and prejudice',
    'Mansfield park',
    'The Tale of Two cities',
    'Great Expectations'
]

book_patterns = list(nlp.pipe(book_names))

# Adding the pattern to the matcher
matcher.add("identify_books", None, *book_patterns)

# You can go ahead and write the function for custom pipeline.
# This function shall use the matcher to find the patterns in the doc , add it to doc.
# ents and return the doc.
# Note that when matcher is applied on a Doc ,
# it returns a tuple containing (match_id,start,end).
# You can extract the span using the start and end indices and store it in doc.ents

from spacy.tokens import Span


def identify_books(doc:spacy.tokens.doc.Doc):
    # Apply the matcher to YOUR doc
    matches = matcher(doc)
    # Create a Span for each match and assign them under label "BOOKS"
    spans = [Span(doc, start, end, label="BOOKS") for match_id, start, end in matches]
    doc.ents = spans
    return doc


nlp.add_pipe(identify_books,after='ner')
print(nlp.pipe_names)


doc = nlp("The library has got several new copies of Mansfield park and Great Expectations . "
          "I have filed a suggestion to buy more copies of The Tale of Two cities ")

print([(ent.text,ent.label_) for ent in doc.ents])