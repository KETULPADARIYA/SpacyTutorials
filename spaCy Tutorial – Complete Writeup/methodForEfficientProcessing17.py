from timeit import timeit

import spacy

from loadSpacy1 import nlp

list_of_text_data = [
    'In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.',
    'Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.',
    'Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving',
    'As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.',
    'The term military simulation can cover a wide spectrum of activities, ranging from full-scale field-exercises,[2] to abstract computerized models that can proceed with little or no human involvement',
    'As a general scientific principle, the most reliable data comes from actual observation and the most reliable theories depend on it.[4] This also holds true in military analysis',
    'Any form of training can be regarded as a "simulation" in the strictest sense of the word (inasmuch as it simulates an operational environment); however, many if not most exercises take place not to test new ideas or models, but to provide the participants with the skills to operate within existing ones.',
    'ull-scale military exercises, or even smaller-scale ones, are not always feasible or even desirable. Availability of resources, including money, is a significant factor—it costs a lot to release troops and materiel from any standing commitments, to transport them to a suitable location, and then to cover additional expenses such as petroleum, oil and lubricants (POL) usage, equipment maintenance, supplies and consumables replenishment and other items',
    'Moving away from the field exercise, it is often more convenient to test a theory by reducing the level of personnel involvement. Map exercises can be conducted involving senior officers and planners, but without the need to physically move around any troops. These retain some human input, and thus can still reflect to some extent the human imponderables that make warfare so challenging to model, with the advantage of reduced costs and increased accessibility. A map exercise can also be conducted with far less forward planning than a full-scale deployment, making it an attractive option for more minor simulations that would not merit anything larger, as well as for very major operations where cost, or secrecy, is an issue']

# %%timeit
get_doc = """
def get_doc():
    global list_of_text_data
    return  [nlp(text) for text in list_of_text_data]
"""


# print(exec('get_doc()'))

def get_time(get_doc):
    print(f"{timeit(stmt=get_doc, number=10000) * 1000:2.4f} ms")


get_time(get_doc)

get_doc2 = """
def get_doc():
    global list_of_text_data
    return  list(nlp.pipe(list_of_text_data))
"""
get_time(get_doc2)

# docs = [nlp(text) for text in list_of_text_data]


nlp.remove_pipe("tagger")
print(nlp.pipe_names)

get_time(get_doc2)

nlp.remove_pipe("parser")
print(nlp.pipe_names)
get_time(get_doc2)

# disable pipeline components in spaCy
nlp = spacy.load("en_core_web_sm",disable = ['tagger','parser'])
print(nlp.has_pipe("tagger"))
print(nlp.has_pipe("parser"))


# to disable pipe lines for some task

# 1st
nlp = spacy.load("en_core_web_sm")
for doc in nlp.pipe(list_of_text_data,disable = ["ner","parser"]):
    print(doc,"\n",f' doc is_tagged {doc.is_tagged} , doc is_parsed {doc.is_parsed}'
                   f', is_nered {doc.is_nered}')

nlp = spacy.load("en_core_web_sm")

with nlp.disable_pipes("tagger","ner"):
    print("-- Inside the block -- ")
    doc =  nlp("The pandemic has disrupted the lives of may")
    print(doc.is_nered)

print("-- Outside the block --")
doc = nlp("I will be tagged and parsed.")
print(f" Is doc nered ? ans > {doc.is_nered}")