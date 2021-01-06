from pprint import pprint

from loadSpacy1 import nlp

mobile_industry_article="""30 Major mobile phone brands Compete in India – A Case Study of Success and Failures Is 
the Indian mobile market a terrible War Zone? We have more than 30 brands competing with each other. Let’s find out 
some insights about the world second-largest mobile bazaar.There is a massive invasion by Chinese mobile brands in 
India in the last four years. Some of the brands have been able to make a mark while others like Meizu, Coolpad, ZTE, 
and LeEco are a failure.On one side, there are brands like Sony or HTC that have quit from the Indian market on the 
other side we have new brands like Realme or iQOO entering the marketing in recent months.The mobile market is so 
competitive that some of the brands like Micromax, which had over 18% share back in 2014, now have less than 5%. Even 
the market leader Samsung with a 34% market share in 2014, now has a 21% share whereas Xiaomi has become a market 
leader. The battle is fierce and to sustain and scale-up is going to be very difficult for any new entrant.new comers 
in Indian Mobile MarketiQOO –They have recently (March 2020) launched the iQOO 3 in India with its first 5G phone – 
iQOO 3. The new brand is part of the Vivo or the BBK electronics group that also owns several other brands like Oppo, 
Oneplus and Realme.Realme – Realme launched the first-ever phone – Realme 1 in November 2018 and has quickly became a 
popular brand in India. The brand is one of the highest sellers in online space and even reached a 16% market share 
threatening Xiaomi’s dominance.iVoomi – In 2017, we have seen the entry of some new Chinese mobile brands likeiVoomi 
which focuses on the sub 10k price range, and is a popular online player. They have an association with 
Flipkart.Techno &amp; Infinix – Transsion Group’s Tecno and Infinix brands debuted in India in mid-2017 and are 
focusing on the low end and mid-range phones in the price range of Rs. 5000 to Rs. 12000.10.OR &amp; Lephone – 10.OR 
has a partnership with Amazon India and is an exclusive online brand with phones like 10.OR D, G and E. However, 
the brand is not very aggressive currently.Kult – Kult is another player who launched a very aggressively priced Kult 
Beyond mobile in 2017 and followed up by launching 2-3 more models.However, most of these new brands are finding it 
difficult to strengthen their footing in India. As big brands like Xiaomi leave no stone unturned to make things 
difficult.Also, it is worth noting that there is less Chinese players coming to India now. As either all the big 
brands have already set shop or burnt their hands and retreated to the homeland China.Chinese/ Global  Brands Which 
failed or are at the Verge of Failing in India? There are a lot more failures in the market than the success stories. 
Let’s first look at the failures and then we will also discuss why some brands were able to succeed in India.HTC – 
The biggest surprise this year for me was the failure of HTC in India. The brand has been in the country for many 
years, in fact, they were the first brand to launch Android mobiles. Finally HTC decided to call it a day in July 
2018.LeEco – LeEco looked promising and even threatening to Xiaomi when it came to India. The company launched a 
series of new phones and smart TVs at affordable rates. Unfortunately, poor financial planning back home caused the 
brand to fail in India too.LG – The company seems to have lost focus and are doing poorly in all segments. While the 
budget and mid-range offering are uncompetitive, the high-end models are not preferred by buyers.Sony – Absurd 
pricing and lack of ability to understand the Indian buyers have caused Sony to shrink mobile operations in India. In 
the last 2 years, there are far fewer launches and hardly any promotions or hype around the new products.Meizu – 
Meizu is also a struggling brand in India and is going nowhere with the current strategy. There are hardly any 
popular mobiles nor a retail presence.ZTE – The company was aggressive till last year with several new phones 
launching under the Nubia banner, but with recent issues in the US, they have even lost the plot in India.Coolpad – I 
still remember the first meeting with Coolpad CEO in Mumbai when the brand started operations. There were big dreams 
and ambitions, but the company has not been able to deliver and keep up with the rivals in the last 1 year.Gionee – 
Gionee was doing well in the retail, but the infighting in the company and loss of focus from the Chinese parent 
company has made it a failure. The company is planning a comeback. However, we will have to wait and see when that 
happens. """



mobile_doc = nlp(mobile_industry_article)
list_of_org = []

for entity in mobile_doc.ents:
    if entity.label_ == "ORG":
        list_of_org.append(entity.text)

pprint(list_of_org)