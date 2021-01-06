# to train NER models by both updating an existing spacy model to suit the specific context of your text documents and
# also to train a fresh NER model from scratch.
import re
from pathlib import Path
from random import shuffle

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import minibatch, compounding, compile_prefix_regex, compile_suffix_regex

nlp = spacy.load("en_core_web_sm")


def modify_str(raw_str: str) -> str:
    """
        Too remove specialize characters in raw_str
    Args:
        raw_str (str): text
    """
    symbols = {",", " ", r'/', "+", "_", "-", ".", "(", ")", '"', "'"}
    characters = []
    for character in raw_str:
        if character.isalnum():
            characters.append(character)
        elif character in symbols:
            characters.append(" " + character + " ")
        elif character in {"\n"}:
            characters.append(" ")
    raw_str = "".join(characters)

    return " ".join(raw_str.split())


text = 'Product # 136-05\r\n\r\nThe Lowrance Navico B744V is a B744V bronze thru hull transducer featuring depth, ' \
       'speed and temperature sensors in a single housing-drill. It operates on dual frequency of 50/200 kHz, ' \
       'with single-ceramic element providing good target detail in both shallow and deep waters.\r\n\n\nNavico ' \
       'B744V Features:\n\n\n\nDual Frequency Transducer\n600 W RMS Power\nHigh Speed Fairing Block\nOperating ' \
       'Frequency of 50/200 kHz\nProvides Depth, Temp & Speed Data\nMid-Performance\nBronze Housing\n7-Pin Blue ' \
       'Connector\n10 Meter Cable Attached\n '

print(modify_str(text))

prefix_re = re.compile(r'''[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']''')
infix_re = spacy.util.compile_infix_regex((r'''[-~]''', r'''(?<=[0-9])(?=[A-Za-z])'''))


def create_tokenizer(nlp):
    return Tokenizer(nlp.vocab,
                     rules={},
                     prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer
                     )


list(nlp(u'Drink 8cups water every-day.'))

def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

nlp = spacy.load('en')
nlp.tokenizer = custom_tokenizer(nlp)

doc = nlp(u'Note: Since the fourteenth century the practice of “medicine” has become a profession; and more importantly, it\'s a male-dominated profession.')
[token.text for token in doc]