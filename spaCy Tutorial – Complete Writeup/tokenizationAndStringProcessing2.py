from loadSpacy1 import nlp, my_text

my_doc = nlp(my_text)
for word in my_doc:
    print(word, end="\t")
print()

# What is the need for Text Preprocessing ?
#
# The outcome of the NLP task you perform, be it classification, finding sentiments, topic modeling etc, the quality
# of the output depends heavily on the quality of the input text used.
#
# Stop words and punctuation usually (not always) don’t add value to the meaning of the text and can potentially
# impact the outcome. To avoid this, its might make sense to remove them and clean the text of unwanted characters
# can reduce the size of the corpus.
#

for token in my_doc:
    print(token.text, ' is_stop word :', token.is_stop, ' is_punctuation :  ', token.is_punct)


def get_cleaned_token(doc, verbose=False):
    a = [token for token in doc if not any(token.is_punct, token.is_stop)]

    if verbose:
        print(f"length of doc is {len(doc)}")
        print(f"length of cleaned doc is {len(a)}")

    return a


for token in get_cleaned_token(my_doc, True):
    print(token.text)
