from pprint import pprint

import pandas as pd
import spacy
from spacy.gold import minibatch
from spacy.util import compounding

from trainTextClassificationModel.prepareTrainingData import get_training_tuple, split_data, evaluate

reviews = pd.read_csv("E-Commerce%20Reviews.csv")

# Extract desired columns and view the dataframe
# need only to column Review Text and Recommend Ind

reviews = reviews[['Review Text','Recommended IND']].dropna()
print(reviews.head(10))





nlp = spacy.load("en_core_web_sm")

# nlp has three components ['tagger','parser','ner']
print(nlp.pipe_names)

# nlp doesn't have a text classifier. so let's just add the built in textcat
# pipeline component of spacy for text classification to our pipeline.

textcat = nlp.create_pipe("textcat",config={"exclusive_classes":True,
                                            'architecture':'simple_cnn'})
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

nlp.add_pipe(textcat,last=True)

# nlp has three components ['tagger','parser','ner','textcat']
print(nlp.pipe_names)

# add desire label to pipeline component.

data = get_training_tuple(reviews)



n_iter = 10
n_texts = 2348


# The data is stored in train_data() variable and the component in texcat.
#
# Before we train, you need to disable the other pipeline components except textcat.
# This is to prevent the other components from getting affected while training.
# It can be accomplished through the disable_pipes() method.
# Next, use the begin_training() function that will return you an optimizer .

(train_texts, train_cats), (dev_texts, dev_cats) = split_data(data,limit=n_texts)
train_data = list(zip(train_texts,[{'cats': cats} for cats in train_cats]))

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]


# A good model should have a good precision as well as a high recall. So ideally, I want to have a measure that
# combines both these aspects in one single metric – the F1 Score.


# nlp.update commands
# docs: This expects a batch of texts as input. You can pass each batch to the zip method,
#       which will return you batches of text and annotations.
# golds: You can pass the annotations we got through the zip method here
# drop: This represents the dropout rate.
# losses: A dictionary to hold the losses against each pipeline component.
#         Create an empty dictionary and pass it here.

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS","PRECISION","RECALL","F_SCORE"))

    # Performing training
    for i in range(n_iter):
        losses = {}
        batches = minibatch(train_data,size=compounding(4.,32.,1.001))


        for batch in batches:
            texts,annotations = zip(*batch)

            nlp.update(texts,annotations,sgd = optimizer,drop = 0.2,losses=losses)

        # Calling with evaluate function and printing score
        with textcat.model.use_params(optimizer.averages):
            scores = evaluate(nlp.tokenizer,textcat,dev_texts,dev_cats)
        print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
              .format(losses['textcat'], scores['textcat_p'],
                      scores['textcat_r'], scores['textcat_f']))


