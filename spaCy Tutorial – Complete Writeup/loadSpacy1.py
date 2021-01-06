import spacy


def load_spacy():
    return spacy.load("en_core_web_sm")


# Parse text through the `nlp` model
nlp = load_spacy()

my_text = """The economic situation of the country is on edge , as the stock 
    market crashed causing loss of millions. Citizens who had their main investment 
    in the share-market are facing a great loss. Many companies might lay off 
    thousands of people to reduce labor cost"""
doc = nlp(my_text)

if __name__ == '__main__':
    print(type(doc))
    print(dir(doc))

# what exactly is a Doc object ?

# It is a sequence of tokens that contains not just the original text but all the results produced by the spaCy model
# after processing the text. Useful information such as the lemma of the text, whether it is a stop word or not,
# named entities, the word vector of the text and so on are pre-computed and readily stored in the Doc object.
