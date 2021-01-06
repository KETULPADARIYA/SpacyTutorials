# PhraseMatcher which can be used when you have a large number of terms(single or multi-tokens) to be matched in a
# text document. Writing patterns for Matcher is very difficult in this case. PhraseMatcher solves this problem .


# The procedure to use PhraseMatcher is very similar to Matcher.
#
#   1.  Initialize a PhraseMatcher object with a vocab.
#   2.  Define the terms you want to match
#   3.  Add the pattern to the matcher
#   4.   Run the text through the matcher to extract the matching positions.
from spacy.matcher.matcher import Matcher
from spacy.matcher.phrasematcher import PhraseMatcher

from loadSpacy1 import nlp

matcher = PhraseMatcher(nlp.vocab)
terms_list = ['Bruce Wayne', 'Tony Stark', 'Batman', 'Harry Potter', 'Severus Snape']

patterns = [nlp.make_doc(term) for term in terms_list]

matcher.add("phrase_matcher", None, *patterns)

# Matcher Object
fictional_char_doc = nlp("""Superman (first appearance: 1938)  Created by Jerry Siegal and Joe Shuster for Action Comics #1 (DC Comics).Mickey Mouse (1928)  Created by Walt Disney and Ub Iworks for Steamboat Willie.Bugs Bunny (1940)  Created by Warner Bros and originally voiced by Mel Blanc.Batman (1939) Created by Bill Finger and Bob Kane for Detective Comics #27 (DC Comics).
Dorothy Gale (1900)  Created by L. Frank Baum for novel The Wonderful Wizard of Oz. Later portrayed by Judy Garland in the 1939 film adaptation.Darth Vader (1977) Created by George Lucas for Star Wars IV: A New Hope.The Tramp (1914)  Created and portrayed by Charlie Chaplin for Kid Auto Races at Venice.Peter Pan (1902)  Created by J.M. Barrie for novel The Little White Bird.
Indiana Jones (1981)  Created by George Lucas for Raiders of the Lost Ark. Portrayed by Harrison Ford.Rocky Balboa (1976)  Created and portrayed by Sylvester Stallone for Rocky.Vito Corleone (1969) Created by Mario Puzo for novel The Godfather. Later portrayed by Marlon Brando and Robert DeNiro in Coppola’s film adaptation.Han Solo (1977) Created by George Lucas for Star Wars IV: A New Hope. 
Portrayed most famously by Harrison Ford.Homer Simpson (1987)  Created by Matt Groening for The Tracey Ullman Show, later The Simpsons as voiced by Dan Castellaneta.Archie Bunker (1971) Created by Norman Lear for All in the Family. Portrayed by Carroll O’Connor.Norman Bates (1959) Created by Robert Bloch for novel Psycho.  Later portrayed by Anthony Perkins in Hitchcock’s film adaptation.King Kong (1933) 
Created by Edgar Wallace and Merian C Cooper for the film King Kong.Lucy Ricardo (1951) Portrayed by Lucille Ball for I Love Lucy.Spiderman (1962)  Created by Stan Lee and Steve Ditko for Amazing Fantasy #15 (Marvel Comics).Barbie (1959)  Created by Ruth Handler for the toy company Mattel Spock (1964)  Created by Gene Roddenberry for Star Trek. Portrayed most famously by Leonard Nimoy.
Godzilla (1954) Created by Tomoyuki Tanaka, Ishiro Honda, and Eiji Tsubaraya for the film Godzilla.The Joker (1940)  Created by Jerry Robinson, Bill Finger, and Bob Kane for Batman #1 (DC Comics)Winnie-the-Pooh (1924)  Created by A.A. Milne for verse book When We Were Young.Popeye (1929)  Created by E.C. Segar for comic strip Thimble Theater (King Features).Tarzan (1912) Created by Edgar Rice Burroughs for the novel Tarzan of the Apes.Forrest Gump (1986)  Created by Winston Groom for novel Forrest Gump.  Later portrayed by Tom Hanks in Zemeckis’ film adaptation.Hannibal Lector (1981)  Created by Thomas Harris for the novel Red Dragon. Portrayed most famously by Anthony Hopkins in the 1991 Jonathan Demme film The Silence of the Lambs.
Big Bird (1969) Created by Jim Henson and portrayed by Carroll Spinney for Sesame Street.Holden Caulfield (1945) Created by J.D. Salinger for the Collier’s story “I’m Crazy.”  Reworked into the novel The Catcher in the Rye in 1951.Tony Montana (1983)  Created by Oliver Stone for film Scarface.  Portrayed by Al Pacino.Tony Soprano (1999)  Created by David Chase for The Sopranos. Portrayed by James Gandolfini.
The Terminator (1984)  Created by James Cameron and Gale Anne Hurd for The Terminator. Portrayed by Arnold Schwarzenegger.Jon Snow (1996)  Created by George RR Martin for the novel The Game of Thrones.  Portrayed by Kit Harrington.Charles Foster Kane (1941)  Created and portrayed by Orson Welles for Citizen Kane.Scarlett O’Hara (1936)  Created by Margaret Mitchell for the novel Gone With the Wind. Portrayed most famously by Vivien Leigh 
for the 1939 Victor Fleming film adaptation.Marty McFly (1985) Created by Robert Zemeckis and Bob Gale for Back to the Future. Portrayed by Michael J. Fox.Rick Blaine (1940)  Created by Murray Burnett and Joan Alison for the unproduced stage play Everybody Comes to Rick’s. Later portrayed by Humphrey Bogart in Michael Curtiz’s film adaptation Casablanca.Man With No Name (1964)  Created by Sergio Leone for A Fistful of Dollars, which was adapted from a ronin character in Kurosawa’s Yojimbo (1961).  Portrayed by Clint Eastwood.Charlie Brown (1948)  Created by Charles M. Shultz for the comic strip L’il Folks; popularized two years later in Peanuts.E.T. (1982)  Created by Melissa Mathison for the film E.T.: the Extra-Terrestrial.Arthur Fonzarelli (1974)  Created by Bob Brunner for the show Happy Days. Portrayed by Henry Winkler.)Phillip Marlowe (1939)  Created by Raymond Chandler for the novel The Big Sleep.Jay Gatsby (1925)  Created by F. Scott Fitzgerald for the novel The Great Gatsby.Lassie (1938) Created by Eric Knight for a Saturday Evening Post story, later turned into the novel Lassie Come-Home in 1940, film adaptation in 1943, and long-running television show in 1954.  Most famously portrayed by the dog Pal.
Fred Flintstone (1959)  Created by William Hanna and Joseph Barbera for The Flintstones. Voiced most notably by Alan Reed. Rooster Cogburn (1968)  Created by Charles Portis for the novel True Grit. Most famously portrayed by John Wayne in the 1969 film adaptation. Atticus Finch (1960)  Created by Harper Lee for the novel To Kill a Mockingbird.  (Appeared in the earlier work Go Set A Watchman, though this was not published until 2015)  Portrayed most famously by Gregory Peck in the Robert Mulligan film adaptation. Kermit the Frog (1955)  Created and performed by Jim Henson for the show Sam and Friends. Later popularized in Sesame Street (1969) and The Muppet Show (1976) George Bailey (1943)  Created by Phillip Van Doren Stern (then as George Pratt) for the short story The Greatest Gift. Later adapted into Capra’s It’s A Wonderful Life, starring James Stewart as the renamed George Bailey. Yoda (1980) Created by George Lucas for The Empire Strikes Back. Sam Malone (1982)  Created by Glen and Les Charles for the show Cheers.  Portrayed by Ted Danson. Zorro (1919)  Created by Johnston McCulley for the All-Story Weekly pulp magazine story The Curse of Capistrano.Later adapted to the Douglas Fairbanks’ film The Mark of Zorro (1920).Moe, Larry, and Curly (1928)  Created by Ted Healy for the vaudeville act Ted Healy and his Stooges. Mary Poppins (1934)  Created by P.L. Travers for the children’s book Mary Poppins. Ron Burgundy (2004)  Created by Will Ferrell and Adam McKay for the film Anchorman: The Legend of Ron Burgundy.  Portrayed by Will Ferrell. Mario (1981)  Created by Shigeru Miyamoto for the video game Donkey Kong. Harry Potter (1997)  Created by J.K. Rowling for the novel Harry Potter and the Philosopher’s Stone. The Dude (1998)  Created by Ethan and Joel Coen for the film The Big Lebowski. Portrayed by Jeff Bridges.
Gandalf (1937)  Created by J.R.R. Tolkien for the novel The Hobbit. The Grinch (1957)  Created by Dr. Seuss for the story How the Grinch Stole Christmas! Willy Wonka (1964)  Created by Roald Dahl for the children’s novel Charlie and the Chocolate Factory. The Hulk (1962)  Created by Stan Lee and Jack Kirby for The Incredible Hulk #1 (Marvel Comics) Scooby-Doo (1969)  Created by Joe Ruby and Ken Spears for the show Scooby-Doo, Where Are You! George Costanza (1989)  Created by Larry David and Jerry Seinfeld for the show Seinfeld.  Portrayed by Jason Alexander.Jules Winfield (1994)  Created by Quentin Tarantino for the film Pulp Fiction. Portrayed by Samuel L. Jackson. John McClane (1988)  Based on the character Detective Joe Leland, who was created by Roderick Thorp for the novel Nothing Lasts Forever. Later adapted into the John McTernan film Die Hard, starring Bruce Willis as McClane. Ellen Ripley (1979)  Created by Don O’cannon and Ronald Shusett for the film Alien.  Portrayed by Sigourney Weaver. Ralph Kramden (1951)  Created and portrayed by Jackie Gleason for “The Honeymooners,” which became its own show in 1955.Edward Scissorhands (1990)  Created by Tim Burton for the film Edward Scissorhands.  Portrayed by Johnny Depp.Eric Cartman (1992)  Created by Trey Parker and Matt Stone for the animated short Jesus vs Frosty.  Later developed into the show South Park, which premiered in 1997.  Voiced by Trey Parker.
Walter White (2008)  Created by Vince Gilligan for Breaking Bad.  Portrayed by Bryan Cranston. Cosmo Kramer (1989)  Created by Larry David and Jerry Seinfeld for Seinfeld.  Portrayed by Michael Richards.Pikachu (1996)  Created by Atsuko Nishida and Ken Sugimori for the Pokemon video game and anime franchise.Michael Scott (2005)  Based on a character from the British series The Office, created by Ricky Gervais and Steven Merchant.  Portrayed by Steve Carell.Freddy Krueger (1984)  Created by Wes Craven for the film A Nightmare on Elm Street. Most famously portrayed by Robert Englund.
Captain America (1941)  Created by Joe Simon and Jack Kirby for Captain America Comics #1 (Marvel Comics)Goku (1984)  Created by Akira Toriyama for the manga series Dragon Ball Z.Bambi (1923)  Created by Felix Salten for the children’s book Bambi, a Life in the Woods. Later adapted into the Disney film Bambi in 1942.Ronald McDonald (1963) Created by Williard Scott for a series of television spots.Waldo/Wally (1987) Created by Martin Hanford for the children’s book Where’s Wally? (Waldo in US edition) Frasier Crane (1984)  Created by Glen and Les Charles for Cheers.  Portrayed by Kelsey Grammar.Omar Little (2002)  Created by David Simon for The Wire.Portrayed by Michael K. Williams.
Wolverine (1974)  Created by Roy Thomas, Len Wein, and John Romita Sr for The Incredible Hulk #180 (Marvel Comics) Jason Voorhees (1980)  Created by Victor Miller for the film Friday the 13th. Betty Boop (1930)  Created by Max Fleischer and the Grim Network for the cartoon Dizzy Dishes. Bilbo Baggins (1937)  Created by J.R.R. Tolkien for the novel The Hobbit. Tom Joad (1939)  Created by John Steinbeck for the novel The Grapes of Wrath. Later adapted into the 1940 John Ford film and portrayed by Henry Fonda.Tony Stark (Iron Man) (1963)  Created by Stan Lee, Larry Lieber, Don Heck and Jack Kirby for Tales of Suspense #39 (Marvel Comics)Porky Pig (1935)  Created by Friz Freleng for the animated short film I Haven’t Got a Hat. Voiced most famously by Mel Blanc.Travis Bickle (1976)  Created by Paul Schrader for the film Taxi Driver. Portrayed by Robert De Niro.
Hawkeye Pierce (1968)  Created by Richard Hooker for the novel MASH: A Novel About Three Army Doctors.  Famously portrayed by both Alan Alda and Donald Sutherland. Don Draper (2007)  Created by Matthew Weiner for the show Mad Men.  Portrayed by Jon Hamm. Cliff Huxtable (1984)  Created and portrayed by Bill Cosby for The Cosby Show. Jack Torrance (1977)  Created by Stephen King for the novel The Shining. Later adapted into the 1980 Stanley Kubrick film and portrayed by Jack Nicholson. Holly Golightly (1958)  Created by Truman Capote for the novella Breakfast at Tiffany’s.  Later adapted into the 1961 Blake Edwards films starring Audrey Hepburn as Holly. Shrek (1990)  Created by William Steig for the children’s book Shrek! Later adapted into the 2001 film starring Mike Myers as the titular character. Optimus Prime (1984)  Created by Dennis O’Neil for the Transformers toy line.Sonic the Hedgehog (1991)  Created by Naoto Ohshima and Yuji Uekawa for the Sega Genesis game of the same name.Harry Callahan (1971)  Created by Harry Julian Fink and R.M. Fink for the movie Dirty Harry.  Portrayed by Clint Eastwood.Bubble: Hercule Poirot, Tyrion Lannister, Ron Swanson, Cercei Lannister, J.R. Ewing, Tyler Durden, Spongebob Squarepants, The Genie from Aladdin, Pac-Man, Axel Foley, Terry Malloy, Patrick Bateman
Pre-20th Century: Santa Claus, Dracula, Robin Hood, Cinderella, Huckleberry Finn, Odysseus, Sherlock Holmes, Romeo and Juliet, Frankenstein, Prince Hamlet, Uncle Sam, Paul Bunyan, Tom Sawyer, Pinocchio, Oliver Twist, Snow White, Don Quixote, Rip Van Winkle, Ebenezer Scrooge, Anna Karenina, Ichabod Crane, John Henry, The Tooth Fairy,
Br’er Rabbit, Long John Silver, The Mad Hatter, Quasimodo """)

character_matches = matcher(fictional_char_doc)
print(character_matches)


def print_matched_str(doc, matcher_):
    print("\n Matching start")
    if isinstance(matcher_, PhraseMatcher) or isinstance(matcher_, Matcher):
        matcher_ = matcher_(doc)
    for id, st, en in matcher_:
        print(doc.vocab.strings[id],id,doc[st:en])


print_matched_str(fictional_char_doc, character_matches)
# print_matched_str(fictional_char_doc,matcher)


case_insensitive_matcher = PhraseMatcher(nlp.vocab,attr = "LOWER")
my_doc = nlp("I wish to visit new york city")
terms = ["New York"]
pattern = [nlp(term) for term in terms]

case_insensitive_matcher.add('matcher',None,*pattern)

print_matched_str(doc=my_doc, matcher_=case_insensitive_matcher)



my_doc = nlp('From 8 am , Mr.X will be speaking on your favorite channel 191.1. Afterward there shall be an exclusive '
             'interview with actor Vijay on channel 194.1 . Hope you are having a great day. Call us on 666666')

pattern = nlp('154.6')

pin_code_matcher = PhraseMatcher(nlp.vocab, attr ="SHAPE")
pin_code_matcher.add('pin_code_matching', None, pattern)
print_matched_str(my_doc,pin_code_matcher)


