# Creating a doc on news articles
from loadSpacy1 import nlp

news_text="""Indian man has allegedly duped nearly 50 businessmen in the UAE of USD 1.6 million and fled the country 
in the most unlikely way -- on a repatriation flight to Hyderabad, according to a media report on Saturday.Yogesh 
Ashok Yariava, the prime accused in the fraud, flew from Abu Dhabi to Hyderabad on a Vande Bharat repatriation flight 
on May 11 with around 170 evacuees, the Gulf News reported.Yariava, the 36-year-old owner of the fraudulent Royal 
Luck Foodstuff Trading, made bulk purchases worth 6 million dirhams (USD 1.6 million) against post-dated cheques from 
unsuspecting traders before fleeing to India, the daily said. The bought goods included face masks, hand sanitisers, 
medical gloves (worth nearly 5,00,000 dirhams), rice and nuts (3,93,000 dirhams), tuna, pistachios and saffron (3,00,
725 dirhams), French fries and mozzarella cheese (2,29,000 dirhams), frozen Indian beef (2,07,000 dirhams) and halwa 
and tahina (52,812 dirhams). The list of items and defrauded persons keeps getting longer as more and more victims 
come forward, the report said. The aggrieved traders have filed a case with the Bur Dubai police station. The traders 
said when the dud cheques started bouncing they rushed to the Royal Luck's office in Dubai but the shutters were 
down, even the fraudulent company's warehouses were empty. """

def remove_details(word):
    word_ent_type = word.ent_type
    if word_ent_type == "PERSON" or word_ent_type == 'ORG' or word_ent_type == 'GPE':
        return 'UNKNOWN'
    return word.string


# Function where each token of spacy doc is passed through remove_deatils()
def update_article(doc):
   # iterrating through all entities
   print([ i for i in doc])

   for ent in doc.ents:
       ent.merge()
   print([ i for i in doc])
   # Passing each token through remove_details() function.
   tokens = map(remove_details,doc)
   return ''.join(tokens)


# Passing our news_doc to the function update_article()
news_doc = nlp(news_text)
update_article(news_doc)
