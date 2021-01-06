from spacy.matcher.matcher import Matcher

from loadSpacy1 import nlp

matcher = Matcher(nlp.vocab)

my_pattern = [{"LOWER":'version'},{"IS_PUNCT":True},{'LIKE_NUM':True}]



matcher.add('VersionFinder',None,my_pattern)


# Run the Token Matcher
my_text = 'The version : 6 of the app was released about a year back and was not very sucessful. As a comeback, ' \
          'six months ago, version : 7 was released and it took the stage. After that , the app has has the limelight ' \
          'till now. On interviewing some sources, we get to know that they have outlined visiond till version : 12 ,' \
          'the Ultimate. '
my_doc= nlp(my_text)

desired_matches = matcher(my_doc)
print(desired_matches)


for match_id,start,end in desired_matches:
    string_id = nlp.vocab.strings[match_id]
    span = my_doc[start:end]
    print(string_id,span)


text = """I was planning a trip of  Manali town last time. Around same budget trips ? "
    I was visiting Ladakh this summer "
    I have planned visiting NewYork and other abroad places for next year"
    Have you ever visited Kodaikanal?,  Surat and Vadodara are  the cleanest cities in India ."""

doc = nlp(text)

matcher = Matcher(nlp.vocab)
my_pattern = [{"LEMMA":"visit"},{"POS":"PROPN"}]

matcher.add("Visiting_places",None,my_pattern)
matches = matcher(doc)

print(" matches found:",len(matches))

for match_id,start,end in matches:
    print('Match Found',doc[start:end])