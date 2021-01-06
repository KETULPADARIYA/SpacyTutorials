import spacy

from loadSpacy1 import nlp

my_text = 'John plays basketball,if time permits. He played in high school too.'
my_doc = nlp(my_text)

for token in my_doc:
    print(token.text, '---- ', token.pos_, spacy.explain(token.pos_))

# Consider you have a text document of reviews or comments on a post. Apart from genuine words, there will be certain
# junk like “etc” which do not mean anything. How can you remove them ?
#
# Using spacy’s pos_ attribute, you can check if a particular token is junk through token.pos_ == 'X' and remove
# them. Below code demonstrates the same.

# Raw text document
raw_text = """I liked the movies etc The movie had good direction  The movie was amazing i.e.
            The movie was average direction was not bad The cinematography was nice. i.e.
            The movie was a bit lengthy  otherwise fantastic  etc etc"""

# Creating a spacy object
raw_doc = nlp(raw_text)

# Checking pos tag is X and printing them
print("The junk values are ..")
for token in raw_doc:
    if token.pos_ == "X":
        print("junks", token.text)

# Remove junks
clean_doc = [ token for token in raw_doc if not token.pos_ == "X"]
print(clean_doc)

# get all tags
all_tags = { token.pos:token.pos_ for token in raw_doc}
print(all_tags)


from spacy import  displacy
my_text = "She never like playing , reading was her hobby"
my_doc = nlp(my_text)

displacy.render(my_doc,style = 'dep')

