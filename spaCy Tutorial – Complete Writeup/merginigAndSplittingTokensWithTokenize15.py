# Printing tokens of a text
from loadSpacy1 import nlp

text = "John Wick is a 2014 American action thriller film directed by Chad Stahelski"
doc = nlp(text)


def print_token(doc):
    print('\n ---- tokens')
    for token in doc:
        print(token.text, token.pos_)


print_token(doc)
# You can see from the output that ‘John’ and ‘Wick’ have been recognized as separate tokens. Same goes for the
# director’s name “Chad Stahelski”

with doc.retokenize() as retokenizer:
    # attrs : You can use it to set attributes to set on the merged token.
    # Here, I want to set the POS (part of speech tag) for “John Wick” as PROPN.(proper noun).
    #  You can use attrs={"POS" : "PROPN"} to achieve it.
    attrs = {"POS": "PROPN"}

    retokenizer.merge(doc[0:2], attrs=attrs)

print_token(doc)


# Splitting tokens using retokenzier.split()

doc = nlp("I purchased the trendy OnePlus7")
print([ i for i in doc])
with doc.retokenize() as retokenizer:
    heads = [(doc[3],1),doc[2]]
    retokenizer.split(doc[4],['OnePlus',"7"],heads =heads)

print_token(doc)

# nlp.rename_pipe

nlp.rename_pipe(old_name="ner",new_name="my_custom_ner")
print(nlp.pipe_names)


