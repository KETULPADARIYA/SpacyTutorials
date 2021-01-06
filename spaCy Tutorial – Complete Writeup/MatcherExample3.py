from spacy.matcher.matcher import Matcher

from loadSpacy1 import nlp

engineering_text = """If you study 
aeronautical engineering, you could specialize in aerodynamics, aeroelasticity, 
composites analysis, avionics, propulsion and structures and materials. 
If you choose to study
 
chemical engineering,
 you may like to
specialize in  chemical reaction engineering, plant design, process engineering, process design or transport phenomena. Civil engineering is the professional practice of designing and developing infrastructure projects. This can be on a huge scale, such as the development of
nationwide transport systems or water supply networks, or on a smaller scale, such as the development of single roads or buildings.
specializations of 

civil engineering  include structural engineering, architectural engineering, transportation engineering, geotechnical engineering,
environmental engineering and hydraulic engineering. 

Computer engineering concerns the design and prototyping of computing hardware and software. 
This subject merges electrical engineering with computer science, oldest and broadest types of engineering, 

mechanical engineering is concerned with the design, manufacturing and maintenance of mechanical systems. 
You’ll study statics and dynamics, thermodynamics, fluid dynamics, stress analysis, mechanical design and
technical drawing
"""

doc = nlp(engineering_text)
Words = [ 'aeronautical engineering','chemical engineering',
'civil engineering',
'Computer engineering',
'mechanical engineering',]

my_pattern = [{"POS":{"IN":["NOUN","ADJ"]}},{"LOWER":'engineering'}]

matcher  = Matcher(nlp.vocab)

matcher.add("identity course",None,my_pattern)
matches = matcher(doc)
print("Total Matches found:",len(matches))

for match_id,start,end in matches:
    print("Matches found",doc[start:end])