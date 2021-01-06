from loadSpacy1 import nlp

employee_text=""" name : Koushiki age: 45 email : koushiki@gmail.com
                 name : Gayathri age: 34 email: gayathri1999@gmail.com
                 name : Ardra age: 60 email : ardra@gmail.com
                 name : pratham parmar age: 15 email : parmar15@yahoo.com
                 name : Shashank age: 54 email: shank@rediffmail.com
                 name : Utkarsh age: 46 email :utkarsh@gmail.com"""

employee_doc = nlp(employee_text)

for token in employee_doc:
    if token.like_email:
        print(token.text)