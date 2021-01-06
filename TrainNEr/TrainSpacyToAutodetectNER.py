# to train NER models by both updating an existing spacy model to suit the specific context of your text documents and
# also to train a fresh NER model from scratch.
from pathlib import Path
from random import shuffle

import spacy
from spacy.util import minibatch, compounding

nlp = spacy.load("en_core_web_sm")

# Getting the pipeline component
# ner = nlp.get_pipe("ner")

#  4. Format of the training examples
TRAIN_DATA = [
    ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
    ("I reached Chennai yesterday.", {"entities": [(19, 28, "GPE")]}),
    ("I recently ordered a book from Amazon", {"entities": [(24, 32, "ORG")]}),
    ("I was driving a BMW", {"entities": [(16, 19, "PRODUCT")]}),
    ("I ordered this from ShopClues", {"entities": [(20, 29, "ORG")]}),
    ("Fridge can be ordered in Amazon ", {"entities": [(0, 6, "PRODUCT")]}),
    ("I bought a new Washer", {"entities": [(16, 22, "PRODUCT")]}),
    ("I bought a old table", {"entities": [(16, 21, "PRODUCT")]}),
    ("I bought a fancy dress", {"entities": [(18, 23, "PRODUCT")]}),
    ("I rented a camera", {"entities": [(12, 18, "PRODUCT")]}),
    ("I rented a tent for our trip", {"entities": [(12, 16, "PRODUCT")]}),
    ("I rented a screwdriver from our neighbour", {"entities": [(12, 22, "PRODUCT")]}),
    ("I repaired my computer", {"entities": [(15, 23, "PRODUCT")]}),
    ("I got my clock fixed", {"entities": [(16, 21, "PRODUCT")]}),
    ("I got my truck fixed", {"entities": [(16, 21, "PRODUCT")]}),
    ("Flipkart started it's journey from zero", {"entities": [(0, 8, "ORG")]}),
    ("I recently ordered from Max", {"entities": [(24, 27, "ORG")]}),
    ("Flipkart is recognized as leader in market", {"entities": [(0, 8, "ORG")]}),
    ("I recently ordered from Swiggy", {"entities": [(24, 29, "ORG")]})
]


def add_label_from_train_data(TRAIN_DATA,nlp):
    ner = nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:  # _ = "I recently ordered from Swiggy", annotations = {"entities": [(24,29, "ORG")]}
        for ent in annotations.get("entities"):  # get_entities ,ent = (24,29,"ORG")
            ner.add_label(ent[2])  # ent[2]

# disable pipeline components

def train_model(TRAIN_DATA):
    pipe_exceptions = {"ner", "trf_wordpiecer", 'trf_tok2vec'}
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    n_iter = 30

    with nlp.disable_pipes(*unaffected_pipes):
        for _ in range(n_iter):
            shuffle(TRAIN_DATA)

            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4, 32, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of text
                    annotations,  # batch of annotaions
                    drop=0.5,
                    losses=losses

                )
                print("Losses", losses)


# predict on new texts :
doc = nlp("I was driving  a Alto.")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

output_dir = Path("../../content/")


def save_model(output_dir):
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)


# Load the saved model and predict
print("Loading from ", output_dir)
nlp_updated = spacy.load(output_dir)
doc = nlp_updated("Fridge can be ordered in FlipKart")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
