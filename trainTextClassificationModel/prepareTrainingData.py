from random import  shuffle


def get_training_tuple(reviews):
    return reviews.apply(lambda row:(row['Review Text'], row["Recommended IND"]), axis =1).tolist()



def split_data(train_data,limit =0 ,split =0.8):

    # take only number of instances
    if limit:
        train_data = train_data[:limit]

    # shuffle the training data .
    shuffle(train_data)

    texts, labels = zip(*train_data)
    cats = [{"POSITIVES":bool(y),"NEGATIVE":not bool(y)} for y in labels]

    split = int(len(train_data)*split)

    return (texts[:split],cats[:split]),(texts[split:],cats[split:])



#        True
#       +      -
# p   ------------
# r + | TP |  FP |
# e - | FN |  TN |
# d   ------------


def evaluate(tokenizer,textcat,texts,cats):
    docs = ( tokenizer(text) for text in texts)
    tp,fp,fn,tn = 0.0,1e-8,1e-8,0.0

    for index,doc in enumerate(textcat.pipe(docs)):
        gold = cats[index]
        for label,score in doc.cats.items():
            if label not  in gold:
                continue
            if label == "NEGATIVE":
                continue

            if score >= 0.5 : # prediction is true (positive)
                if gold[label] >= 0.5: # original is true
                    tp += 1.0
                else:
                    fp += 1.0
            else: # prediction is false (negative)
                if gold[label] >= 0.5: # original is true
                    tn += 1.0
                else: # original is false
                    fn += 1.0
    precision = tp/ (tp + fp)
    recall = tp/(tp+fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision *recall)/ (precision +recall)
    return  {
        "textcat_p":precision,
        "textcat_r":recall,
        "textcat_f":f_score
    }

