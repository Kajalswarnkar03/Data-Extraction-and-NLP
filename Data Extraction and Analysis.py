#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing library
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen as uReq
import requests
import urllib
import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
#nltk.download('stopwords')
#nltk.download('wordnet')
import string
#nltk.download('punkt')


# In[2]:


#importing input file
df=pd.read_csv('input1.csv')[['URL_ID','URL']]


# In[3]:


df=df.iloc[0:100]


# In[4]:


df


# In[5]:


df.drop('URL_ID',axis=1,inplace=True)


# # Data Extraction

# In[6]:


'''import pandas as pd
import requests
from bs4 import BeautifulSoup

# Ensure you have a DataFrame df with your URLs
# For example:
# df = pd.DataFrame({'urls': ['http://example.com/page1', 'http://example.com/page2']})

url_id = 1
for i in range(len(df)):
    j = df.iloc[i].values

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
    
    # Load text from URL
    page = requests.get(j[0], headers=headers)
    
    # Parse the page content
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Extract text content
    content = soup.findAll(attrs={'class': 'td-post-content'})
    if content:
        content_text = content[0].text.replace('\xa0', " ").replace('\n', " ")
    else:
        content_text = ''
    
    # Extract title
    title = soup.findAll(attrs={'class': 'entry-title'})
    if title:
        title_text = title[0].text.replace('\n', " ").replace('/', "")
    else:
        title_text = ''
    
    # Merge title and content text
    text = title_text + '.' + content_text
    
    # Save the text to a file
    filename = f"{url_id}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)
    
    url_id += 1'''


# # Data analysis

# In[7]:


#Read the Extracted Text Data
import pandas as pd

# Initialize an empty list to store the text data
text_data = []

# Assuming you have saved files named 1.txt, 2.txt, ..., you can read them
num_files = 100  # Adjust based on the number of files you have
for i in range(1, num_files + 1):
   filename = f"{i}.txt"
   with open(filename, 'r', encoding='utf-8') as file:
       text = file.read()
       text_data.append({'id': i, 'text': text})

# Create a DataFrame from the collected text data
df_text = pd.DataFrame(text_data)


# In[8]:


#!pip install textblob


# In[9]:


#Textual Analysis and Variable Computation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Download NLTK resources if not already downloaded
#nltk.download('punkt')
#nltk.download('stopwords')

# Function to compute variables for each text
def compute_variables(text):
    # Tokenization and word count
    tokens = word_tokenize(text)
    word_count = len(tokens)
    
    # Stopword count
    stop_words = set(stopwords.words('english'))
    stopwords_count = sum(1 for word in tokens if word.lower() in stop_words)
    
    # Average word length
    if word_count > 0:
        avg_word_length = sum(len(word) for word in tokens) / word_count
    else:
        avg_word_length = 0
    
    # Sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    return {
        'Word Count': word_count,
        'Stopword Count': stopwords_count,
        'Average Word Length': avg_word_length,
        'Sentiment Polarity': sentiment_polarity,
        'Sentiment Subjectivity': sentiment_subjectivity
    }

# Compute variables for each text in df_text
df_text['Variables'] = df_text['text'].apply(compute_variables)


# In[10]:


'''Blank_link={}
class DataIngestion:
    
    def secondary(self):
        data=pd.read_excel('C:/Users/RANU RAJA/Desktop/Blockcoffer project/Input.xlsx')
        df=data.copy() #create a copy to avoid modifying the original DataFrame

        # Create an empty 'article_words' column
        df['article_words']=''

        for i, url in enumerate(df['URL']):
            response_code=requests.get(url)
            soup=bs(response_code.text, 'html.parser')
            article_title=soup.find('title').text

            all_text_element=soup.find("div", class_="td-post-content tagdiv-type")
            
            if all_text_element is not None:
                all_text=all_text_element.get_text(strip=True, separator='\n')
                firstdata=all_text.splitlines()
            else:
                print(f"No matching element found in the HTML for URL: {url}")
                Blank_link[f"blackassign00{i+1}"]=url
                firstdata=[]
                for i, j in Blank_link.items():
                    if url==j:
                        response_code=requests.get(j)
                        soup=bs(response_code.text,'html.parser')
                        article_title=soup.find('title').text
                        alldiv=soup.find("div", class_="td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type")
                        firstdata=alldiv.text
                        return firstdata
                else:
                       print(f"No mathching element found in the HTML for URL: {url}")

            filename=urllib.parse.quote_plus(url)
            file_path='C:/Users/RANU RAJA/Desktop/Blockcoffer project'
            space=" "

            with open(f"{file_path}\{filename}.txt", 'w+') as file1:
                file1.writelines(article_title)
                file1.writelines(space)
                file1.writelines(firstdata)

            # Update 'article_words' column for the current row 
            df.at[i, 'article_words']=f"{article_title}-{firstdata}"

        df.to_csv('C:/Users/RANU RAJA/Desktop/Blockcoffer project/Input.csv', index=False)
        return df

if __name__ == "__main__":
    obj = DataIngestion()
    obj.secondary()'''


# In[11]:


#df.columns


# In[12]:


#Blank_link


# In[13]:


#file_path='C:/Users/RANU RAJA/Desktop/Blockcoffer internship project'


# In[14]:


'''updated_list=[]

for i, j in Blank_link.items():
    response_code=requests.get(j)
    soup=bs(response_code.text, 'html.parser')
    article_title=soup.find('title').text

    alldiv=soup.find("div", class_="td_block_wrap tdb_single_content tdi_130 td-pb-border-top td_block_template_1 td-post-content tagdiv-type")

    if alldiv is not None:
        firstdata=alldiv.text
        filename=urllib.parse.quote_plus(j)
        filepath='C:/Users/RANU RAJA/Desktop/Blockcoffer internship project'
        space=" "

        with open(f"{file_path}\{filename}.txt", 'w+') as file1:
                file1.writelines(article_title)
                file1.writelines(space)
                file1.writelines(firstdata)
        updated_dict={
            'URL_ID':i,
            'URL':j,
            'article_words':f"{article_title} - {firstdata}"
        }
        updated_list.append(updated_dict)
    else:
        print(f"No data available for the link: {j}")
updated_df=pd.DataFrame(updated_list)'''


# In[15]:


#remain_data=pd.DataFrame(updated_df)


# In[16]:


#remain_data


# In[17]:


#df=pd.merge(df, updated_df[['URL', 'article_words']], on='URL', how='left')


# In[18]:


#df.head()


# In[19]:


# Again try to perform analysis on data and performing operations
text=pd.read_csv('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/100.txt',header=None)
     


# In[20]:


#information of data frame
text.info()


# In[21]:


#removing extra created column
text.drop(1,axis=1,inplace=True)


# In[22]:


#converting type
text=text.astype(str)


# In[23]:


#converting text to sentence
import re
a=text[0].str.split('([\.]\s)',expand=False)#splitting text on '.'
b=a.explode()#converting to rows
b=pd.DataFrame(b)#creating data frame
b.columns=['abc']


# In[24]:


#removing . char from each rows
def abcd(x):    
    nopunc =[char for char in x if char != '.']
    return ''.join(nopunc)
b['abc']=b['abc'].apply(abcd)


# In[25]:


#replacing emty space with null values
c=b.replace('',np.nan,regex=True)
c=c.mask(c==" ")
c=c.dropna()
c.reset_index(drop=True,inplace=True)


# In[26]:


#importing nltk library and stopwords
import nltk
import string


# In[27]:


punc=[punc for punc in string.punctuation]
punc


# In[28]:


#importing stop words files that are provided
StopWords_Auditor=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_Auditor.txt','r',encoding='ISO-8859-1')
StopWords_Currencies=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_Currencies.txt','r',encoding='ISO-8859-1')
StopWords_DatesandNumbers=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_DatesandNumbers.txt','r',encoding='ISO-8859-1')
StopWords_Generic=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_Generic.txt','r',encoding='ISO-8859-1')
StopWords_GenericLong=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_GenericLong.txt','r',encoding='ISO-8859-1')
StopWords_Geographic=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_Geographic.txt','r',encoding='ISO-8859-1')
StopWords_Names=open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/StopWords/StopWords_Names.txt','r',encoding='ISO-8859-1')


# In[29]:


StopWords_Auditor.seek(0)
StopWords_Auditor.readlines()


# In[30]:


#creating func for removing stop words
def text_process(text):
    nopunc =[char for char in text if char not in punc or char not in [':',',','(',')','’','?']]
    nopunc=''.join(nopunc)
    txt=' '.join([word for word in nopunc.split() if word.lower() not in StopWords_Auditor])
    txt1=' '.join([word for word in txt.split() if word.lower() not in StopWords_Currencies])
    txt2=' '.join([word for word in txt1.split() if word.lower() not in StopWords_DatesandNumbers])
    txt3=' '.join([word for word in txt2.split() if word.lower() not in StopWords_Generic])
    txt4=' '.join([word for word in txt3.split() if word.lower() not in StopWords_GenericLong])
    txt5=' '.join([word for word in txt4.split() if word.lower() not in StopWords_Geographic])
    return ' '.join([word for word in txt5.split() if word.lower() not in StopWords_Names])


# In[31]:


#applying func for each row
c['abc']=c['abc'].apply(text_process)


# # Variables
# The definition of each of the variables given in the “Text Analysis.docx” file
# 

# In[32]:


import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob

# Ensure necessary NLTK data packages are downloaded
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Load stopwords and sentiment word lists
stopwords = set(nltk.corpus.stopwords.words('english'))
positive= set(open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/MasterDictionary/positive-words.txt', 'r').read().splitlines())
negative= set(open('C:/Users/RANU RAJA/Desktop/Blockcoffer internship project/MasterDictionary/negative-words.txt', 'r').read().splitlines())


# In[33]:


# Example of creating pandas DataFrames from existing data
positive = pd.DataFrame(positive, columns=['abc'])
negative = pd.DataFrame(negative, columns=['abc'])
    


# In[34]:


# Rename the columns to 'abc'
positive.columns = ['abc']
negative.columns = ['abc']

# Convert the 'abc' column to string type
positive['abc'] = positive['abc'].astype(str)
negative['abc'] = negative['abc'].astype(str)


# In[35]:


#positive list
length=positive.shape[0]
post=[]
for i in range(0,length):
   nopunc =[char for char in positive.iloc[i] if char not in string.punctuation or char != '+']
   nopunc=''.join(nopunc)

   post.append(nopunc)


# In[36]:


#negative list
length=negative.shape[0]
neg=[]
for i in range(0,length):
  nopunc =[char for char in negative.iloc[i] if char not in string.punctuation or char != '+']
  nopunc=''.join(nopunc)
  neg.append(nopunc)
     


# In[37]:


#importing tokenize library
from nltk.tokenize import word_tokenize


# In[232]:


txt_list=[]
length=c.shape[0]
for i in range(0,length):
  txt=' '.join([word for word in c.iloc[i]])
  txt_list.append(txt)


# In[233]:


#tokenization of text
tokenize_text=[]
for i in txt_list:
  
  tokenize_text+=(word_tokenize(i))


# In[234]:


print(tokenize_text)


# In[235]:


len(tokenize_text)


# # 1) POSITIVE SCORE

# In[236]:


positive_score=0
for i in tokenize_text:
  if(i.lower() in post):
    positive_score+=1
print('postive score=', positive_score)


# # 2)NEGATIVE SCORE 

# In[237]:


negative_score=0
for i in tokenize_text:
  if(i.lower() in neg):
    negative_score+=1
print('negative score=', negative_score)


# # 3) POLARITY SCORE

# In[238]:


#Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
Polarity_Score=(positive_score-negative_score)/((positive_score+negative_score)+0.000001)
print('polarity_score=', Polarity_Score)


# # 4) SUBJECTIVITY SCORE

# In[239]:


#Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
subjectiivity_score=(positive_score-negative_score)/((len(tokenize_text))+ 0.000001)
print('subjectivity_score',subjectiivity_score)


# # 5) AVG SENTENCE LENGTH

# In[240]:


length=c.shape[0]
avg_length=[]
for i in range(0,length):
  avg_length.append(len(c['abc'].iloc[i]))
avg_senetence_length=sum(avg_length)/len(avg_length)
print('avg sentence length=', avg_senetence_length)


# # 6) PERCENTAGE OF COMPLEX WORDS

# In[241]:


vowels=['a','e','i','o','u']
import re
count=0
complex_Word_Count=0
for i in tokenize_text:
  x=re.compile('[es|ed]$')
  if x.match(i.lower()):
   count+=0
  else:
    for j in i:
      if(j.lower() in vowels ):
        count+=1
  if(count>2):
   complex_Word_Count+=1
  count=0


# In[242]:


Percentage_of_Complex_words=complex_Word_Count/len(tokenize_text)
print('percentag of complex words= ',Percentage_of_Complex_words)


# # 7) FOG INDEX

# In[243]:


#Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
Fog_Index = 0.4 * (avg_senetence_length + Percentage_of_Complex_words)
print('fog index= ',Fog_Index )


# # 8) AVG NUMBER OF WORDS PER SENTENCE

# In[244]:


length=c.shape[0]
avg_length=[]
for i in range(0,length):
  a=[word.split( ) for word in c.iloc[i]]
  avg_length.append(len(a[0]))
  a=0
#avg
avg_no_of_words_per_sentence=sum(avg_length)/length
print("avg no of words per sentence= ",avg_no_of_words_per_sentence)
     


# # 9) COMPLEX WORD COUNT

# In[245]:


vowels=['a','e','i','o','u']
import re
count=0
complex_Word_Count=0
for i in tokenize_text:
  x=re.compile('[es|ed]$')
  if x.match(i.lower()):
   count+=0
  else:
    for j in i:
      if(j.lower() in vowels ):
        count+=1
  if(count>2):
   complex_Word_Count+=1
  count=0
print('complex words count=',  complex_Word_Count)


# # 10) WORD COUNT

# In[246]:


word_count=len(tokenize_text)
print('word count= ', word_count)


# # 11) SYLLABLE PER WORD

# In[247]:


vowels=['a','e','i','o','u']
import re
count=0
for i in tokenize_text:
  x=re.compile('[es|ed]$')
  if x.match(i.lower()):
   count+=0
  else:
    for j in i:
      if(j.lower() in vowels ):
        count+=1
syllable_count=count
print('syllable_per_word= ',syllable_count)


# # 12) PERSONAL PRONOUNS

# In[248]:


pronouns=['i','we','my','ours','us' ]
import re
count=0
for i in tokenize_text:
  if i.lower() in pronouns:
   count+=1
personal_pronouns=count
print('personal pronouns= ',personal_pronouns )


# # 13) AVG WORD LENGTH

# In[249]:


count=0
for i in tokenize_text:
  for j in i:
    count+=1
avg_word_length=count/len(tokenize_text)
print('avg word= ', avg_word_length)


# In[259]:


data={'positive_score':positive_score,'negative_score':negative_score,'Polarity_Score':Polarity_Score,'subjectiivity_score':subjectiivity_score,'avg_senetence_length':avg_senetence_length,'Percentage_of_Complex_words':Percentage_of_Complex_words,'Fog_Index':Fog_Index,'avg_no_of_words_per_sentence':avg_no_of_words_per_sentence,'complex_Word_Count':complex_Word_Count,'word_count':word_count,'syllable_count':syllable_count,'personal_pronouns':personal_pronouns,'avg_word_length':avg_word_length}


# In[266]:


output_df = pd.read_excel('Output Data Structure.xlsx')

# These are the required parameters 
variables = [positive_score,
            negative_score,
            Polarity_Score,
            subjectiivity_score,
            avg_senetence_length,
            Percentage_of_Complex_words,
            Fog_Index,
            avg_no_of_words_per_sentence,
            complex_Word_Count,
            word_count,
            syllable_count,
            personal_pronouns,
            avg_word_length]

# write the values to the dataframe
for i, var in enumerate(variables):
  output_df.iloc[:,i+2] = var

#now save the dataframe to the disk
output_df.to_csv('Output_Data.csv')


# In[ ]:




