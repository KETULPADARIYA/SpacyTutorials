from loadSpacy1 import nlp

text='she played chess against rita she likes playing chess.'
doc = nlp(text)

for token in doc:
    print(token.lemma_)


doc = nlp("I love travelling")
word_hash = nlp.vocab.strings['travelling']
print(f'travelling word hash {word_hash}')

word_string = nlp.vocab.strings[word_hash]
print(f"From {word_hash} get word {word_string}.")

# Interestingly, a word will have the same hash value irrespective of which document it occurs in or which spaCy
# model is being used.

# Create two different doc with a common word
doc1 = nlp('Raymond shirts are famous')
doc2 = nlp('I washed my shirts ')

def print_word_text_hash_value(doc):
    for word in doc :
        hash_value = nlp.vocab.strings[word.text]
        print(word.text,":",hash_value)


print("---doc1 ---")
print_word_text_hash_value(doc1)

print("---doc2---")
print_word_text_hash_value(doc2)
