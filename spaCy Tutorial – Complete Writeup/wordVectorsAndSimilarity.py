# Identifying similarity of two words or tokens is very crucial .
# It is the base to many everyday NLP tasks like text
# classification , recommendation systems, etc..
# It is necessary to know how similar two sentences are ,
# so they can be grouped in same or opposite category.


import spacy

nlp = spacy.load("en_core_web_md")


def check_token_has_vector(text):
    tokens = nlp(text)

    for token in tokens:
        print(token.text, " : ", token.has_vector)


check_token_has_vector("I am an excellent cook")
check_token_has_vector('I wish to go hogwarts LolXD')

bad = "bad"
terrible = "terrible"


def check_similarity(x, y):
    similarity_score = nlp(x).similarity(nlp(y))
    print(f"{x} is {similarity_score * 100:2.3f} % similar to {y} .\n")

check_similarity(bad,terrible)

pizza = 'pizza'
burger = 'burger'
chair = 'chair'
noodles = 'noodles'

a = [pizza,burger,chair,noodles]
for index,i in enumerate(a):
    for j in a[index+1:]:
        check_similarity(i,j)
