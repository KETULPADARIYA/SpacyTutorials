from loadSpacy1 import nlp

text = '2020 is far worse than 2009 or 2001year'

doc = nlp(text)
for token in doc:
    if token.like_num:
        print(token,"int token".title())

production_text='Production in chennai is 87 %. In Kolkata, produce it as low as 43 %. In Bangalore, production ia as ' \
                'good as 98 %.In mysore, production is average around 78 % '
production_doc = nlp(production_text)

for token in production_doc:
    if token.like_num:
        index_of_next_token = token.i + 1
        next_token = production_doc[index_of_next_token]
        if next_token.text == "%":
            print("It is percentage figure ",token)