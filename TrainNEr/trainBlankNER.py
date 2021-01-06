import spacy

from TrainNEr.TrainSpacyToAutodetectNER import TRAIN_DATA, add_label_from_train_data,train_model
# from trainTextClassificationModel.gettingStartedWithCustomTextClassfication import train_data

nlp = spacy.load("en")

nlp.add_pipe(nlp.create_pipe("ner"))
nlp.begin_training()

add_label_from_train_data(TRAIN_DATA,nlp)


train_model(TRAIN_DATA)

doc = nlp("I was driving  a Alto.")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
