
Sentiment analysis has been using as a tool to cassify response from your user/customer as 'positive' or 'negative' or even 'neutral'. It combines two different disciplines : Natural Language Processing and Text Analysis to extract information from text data. In sentiment analysis, algorithm will learn from labelled example data and predict the label of new /unseen data points. This approach is called supervised learning, as we will train our model with corpus of labelled news.

### Why sentiment analysis?

People have different ways to express their attitudes or opinions / reviews towards your product /event / movie or even for people. These user reviews have potential to build brand authenticity between customers and even to establish trust in product.Most of the companies are trying to evaluate the brand value of a product based on customer reviews. In this process, company can extract customer behaviors like : which products are popular among users, what percentage of your customer dislike your brand, how many users have postive sentiment on their competitors product? These questions ultimately help companies to focus on different aspects of product like : product revamp , new marketing strategy and even for social media outreach plan.

Before use of sentiment analysis algorithms, people were trying to evaluate those user response based on simple practices like : extracting keywords from the content and restricting their findings only on 'what people are talking about?'.
This will never helps you out to answer few important questions 'What people are feeling and thinking about your product?'. That means only the explicit statements/reviews or opinions will have measurable output with your old approach. What would you do with large number of implicit comments?

Even though, it is extremely arduous to determine the actual tone from given text, we can try to develop a more accurate version of our older one.

### What tools we are using?

 - Language : python
 - Scraping library : BeautifulSoup
 - Machine learning library : scikit-learn 
 - Data wrangling : Pandas and Numpy
 - Plotting : matplotlib


In the following blog, I am going to help you to understand the basic norms of natural language processing in order solve the problem of sentiment analysis. I have divided the whole content into following sub-topics:
		1. Problem Statement
		2. Problem Analysis
		3. Data Collection
		4. Model Development
		5. Analysis

####  Problem Statement:

Though there are lots of challenges out there that can be solved using data science, I come up with a very basic problem. Why I am choosing this -- simply because I do not want to waste my energy just by thinking about big problems and also not to waste your energy to read it and forget in couple of hours :). 

PROBLEM: We will classify the sentiment of news titles (Whether positive or negative or neutral.)

#### Problem Analysis:

For this part, we will answer following questions:
	   1. What will be the input for your program?
	   2. What output you are expecting ?
          Whether it is continous value (something like given by regression model?) or   it is a label value (like spam or ham?)
	   3. What sort of problem is this?
          Whether it is supervised or unsupervised learning? Is it a problem of classification or  clustering ? 
	   4. What are the features you should take into account?

	As we already know, the sentiment of any text can be classified into
    three major categories (positive, negative, and neutral), the problem
    is clearly a type of classification. And, we will be using labelled
    data (supervised learning).

	Your ML model will not be able to predict the correctly if you don't
    have enough training data.

	Suppose, you feed following two texts into your algorithm,

	'New government rule aganist people will' -->Negative
	'Government offers funding opportunities for the startups' -->Positive

	What would you expect if the new input text is

	'Tech Companies in Nepal are having problem to raise funds.
    Nevertheless, they are doing great with customer acquisition'

	Your model is not being taught with this kind of complex structure.


#### Data Collection:

After you have your problem analysis, you should only focus couple of your days to gather the related data.

 a. Training / testing data collection

	I have collected data from news portal ekantipur.com for training/testing of our sentiment model.

 b. Unseen data collection
    Unseen data set is collection of documents wihtout label. We will evaluate our final     model based on unseen data.
	In my case, I need to collect news titles from the news portals. I chose our national daily paper (setopati.net) as a data source.



```python
# Here is the sample code that I have used for scraping news site.
# news_spider.py

# This module scrolls through news site (ekantipur.com) and collects news titles.
import time
import csv

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome()

browser.get("http://www.ekantipur.com/eng")
time.sleep(1)

elem = browser.find_element_by_tag_name("body")

# set number of pages be scrolled
no_of_pagedowns = 5

# scroll page (handling infinite page scrolling)
while no_of_pagedowns:
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    no_of_pagedowns -= 1

content = browser.page_source
soup = BeautifulSoup(content, "html5lib")
browser.close()

news_title_containers = soup.find_all(
    "div", attrs={'class': 'display-news-title'})
with open('news_titles.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for each_container in news_title_containers:
        title_link = each_container.find('a')
        news_title = title_link.string.strip()
        writer.writerow([news_title])

```

#### Model development

Before starting on model development, you should be familiar with few terms that will help you to understand the process:

i. Training and Testing

This is the process of applying our model on data whose true classes are already known. Once the model is well trained with those data, we need to compute the proportion of the time our classifier will accurately predict the new data. This process is know as testing.

ii. Bias(underfitting) and Variance(overfitting)

Ideally, whenever we try to develop ML model, we try to capture all the features(information) of the training data and also want our model to generalize the output for unseen input data.

<img src="final_draw.png">
In the diagram above, the linear model is underfit
 - The model doesn't take care of all the information provided (high bias)
 - If you provide new data to this model, it will not change it's behavior(shape) (low variance)


On the other hand, the model with high degree polynomial at the right is overfit
 - The model will take into account all the data points (low bias)
 - The curve will try to change it's shape in order to capture new data points, loosing the generality of the model (high variance)


iii. Cross Validation:

Most of the time, when we are developing ML model we simply split the data randomly into train and test set and feed into our model. While doing so, we may have a high accuracy on that particular chunk of training data set. What if your model perform differently and give lower accuracy while feeding different chuncks of from the sama data points? So, in cross-validation, we will be creating number of train/test splits and measure the accuracy by calculating average on each of the spits.


iv. Learning Curve
	
  Learning curve helps to measure the performance of your model based on variations on your training data supply.

   Why you need to plot learning curve?
   - You can identify the correct spot on your curve where bias and variance is minimized.

I have divided the whole model development process into following subprocesses:

 #### A. Loading data / Split into training and test set

I have divided the collected data by 80/20 for training and testing.



```python
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    'news_corpus/data.csv',names=['text', 'label'],
    dtype={'label': object})
data.head(5)

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business growth in euro zone rises to 32-month high</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fed official says weak data caused by weather, should not slow taper</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Forex - Pound drops to one-month lows against euro</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Europe reaches crunch point on banking union</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10 Things You Need To Know This Morning</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let us remove rows having non label values

data = data[data['label'].map(len) <=2]
data['label'].unique()
```




    array(['1', '-1', '0'], dtype=object)




```python

%matplotlib inline

import matplotlib.pyplot as plt

data['label'].value_counts().plot(kind="bar")
plt.show()

```


![png](sentiment_analysis_files/sentiment_analysis_9_0.png)



```python
# train/test data split
train, test = train_test_split(data, test_size=0.2)

```

#### B. Feature Extraction
  i. We need to convert text data into numerical feature vectors
  
  Most of the algorithms expect numerical data rather than raw (a sequence of symbols). This process is called vectorization -- turning text documents into numerical representation. This whole process comprises of - tokenization, counting and normalization of data ( More specifically called prcess of Bag of Words). The document will be represented by the occurrences of words in the document rather than the relative position of them.
  
Let us have a closer look into vectorization


```python
sample_train_data = ['Dispute delays National Assembly formation process',
              'Country is looking to encourage entrepreneurs and startup process',
              'Airline fuel surcharges to go up from Tuesday'] 

from sklearn.feature_extraction.text import CountVectorizer
# instantiate Vectorizer
vec = CountVectorizer()

# feed/learn the 'vocabulary' of the training data
vec.fit(sample_train_data)
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=None, vocabulary=None)



Looking at the definition above, the `CountVectorizer` simply make whole words lowercase, remove duplicates, and remove single word characters as suggested by regex.


```python
# current fitted vocabulary
vec.get_feature_names()
```




    ['airline',
     'and',
     'assembly',
     'country',
     'delays',
     'dispute',
     'encourage',
     'entrepreneurs',
     'formation',
     'from',
     'fuel',
     'go',
     'is',
     'looking',
     'national',
     'process',
     'startup',
     'surcharges',
     'to',
     'tuesday',
     'up']



Let us create a document-term matrix. This simply describes the occurence of terms collection of documents.


```python
# transform training data into a 'document-term matrix'
sample_train_dtm = vec.transform(sample_train_data)
sample_train_dtm
```




    <3x21 sparse matrix of type '<class 'numpy.int64'>'
    	with 23 stored elements in Compressed Sparse Row format>



Looking at the output, we can see that it creates a matrix with 3 rows and 21 columns.
 - Here, 3 colums because we have three documents all together
 - And, 21 rows because we have 21 terms (learned during fitting step above)


```python
# let's observe the dense matrix
sample_train_dtm.toarray()
```




    array([[0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]])




```python
# Let us see the complete data map using pandas
import pandas as pd
pd.DataFrame(sample_train_dtm.toarray(), columns=vec.get_feature_names())
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline</th>
      <th>and</th>
      <th>assembly</th>
      <th>country</th>
      <th>delays</th>
      <th>dispute</th>
      <th>encourage</th>
      <th>entrepreneurs</th>
      <th>formation</th>
      <th>from</th>
      <th>...</th>
      <th>go</th>
      <th>is</th>
      <th>looking</th>
      <th>national</th>
      <th>process</th>
      <th>startup</th>
      <th>surcharges</th>
      <th>to</th>
      <th>tuesday</th>
      <th>up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>


You can see that most of the feature values are `ZEROs`. Why is that? Because most of the documents can be represented by small sets of words. Suppose, you have collected almost 100 reviews (documents) from a site. Then, there could be all together 1000 unique words (corpus) and a single review (single document) might have 10 unique words. right?

Storing such a big number of `ZEROs` does not make sense. So, scikit-learn internally uses `scipy.sparse` package to store such a matrix in memory and it also speed up the operations.


```python
# let us now test our model with new example 
sample_test_data = ['Country is looking to encourage agriculture schemes']
```


```python
# transform testing data into a document-term matrix (using existing vocabulary)
sample_test_dtm = vec.transform(sample_test_data)
sample_test_dtm.toarray()
```




    array([[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]])




```python
# examine the vocabulary and document-term matrix together
pd.DataFrame(sample_test_dtm.toarray(), columns=vec.get_feature_names())
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline</th>
      <th>and</th>
      <th>assembly</th>
      <th>country</th>
      <th>delays</th>
      <th>dispute</th>
      <th>encourage</th>
      <th>entrepreneurs</th>
      <th>formation</th>
      <th>from</th>
      <th>...</th>
      <th>go</th>
      <th>is</th>
      <th>looking</th>
      <th>national</th>
      <th>process</th>
      <th>startup</th>
      <th>surcharges</th>
      <th>to</th>
      <th>tuesday</th>
      <th>up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



You can see that `agriculture` and  `schemes` are not the part of output. For our model, to make prediction about this new observation, this new data must have the same features as the training observations. We did not train on the features "`agriculture` and  `schemes`" so our model would not be able to predict. Make any sense ?

##### Problem with `CountVectorizer` **

If are looking at two different documents talking about same topic, one bit longer than
other, you will find the average count values in the longer one will be higher than
the shorter document. That means, `CountVectorizer` simply gives equal weightage to each
of the word present in your document (one word represents one column in DataFrame). And,
when you have new document, it simply maps 1 if the word is present in that document
otherwise map to 0.

So as to avoid this problem, we can use `TfidfTransformer` which evaluates frequencies of 
words rather than their occurences in the document and helps model to figure out which of
the word is more important in the document and also with respect to the corpus.


#### Let's  take a closer look at `Tfidf`

Suppose you have a corpus of documents talking about `Mountains`. Now, `Tfidf` will work in three different parts

#### i. Term Frequency (Tf)
  It simply looks for the number of times a particular term occurs in your single document.
  Let's take an example. You can find word `is` in almost all of your documents. Do you think the word `is` has any credit to describe your document ? But, at the same time, if you find word `Everest` in the document, you can figure out that the document is about `Mountains`.
 So, we need to find a way to reduce the weightage of word `is`. One of the method is calculating reciprocal of the count value, which minimizes the significant of word `is` in the document. But, you might face another problem here. Suppose, you encounter a word in your document (let's say `stratovolcano` having rare occurence in your document). If you take reciprocal of this occurence, you will get 1 (or close to 1 - hightest weightage). But, this word does not give much sense about `Montains`.
 
For now, we can keep the word count as follows:

    `tf("is") = 1000`
    `tf("Everest") = 50`
    `tf("stratovolcano") = 2`


#### ii. Inverser Document Frequency (iDF)

  The problem of occurence of rare and more frequent words will be handled by this method.
   Inverser Document Frequency gives downscale weights for words that occur in many documents in the corpus by taking log of number of total documents in your corpus divided by the total number of documents having occurence of the word.
   i.e iDF = log (total number of documents/ total documents with word occurence)
   
   Let's calculate iDF for above words. (Suppose we have total 20 documents)
   
    `iDF("is") = log(20/20) = 0` , Since 'is' occurs in all the documents
    `iDF("Everest") = log(20/5) = 0.6`,  Since corpus is talking about 'Mountains'
    `iDF("stratovolcano") = log(10/1) = 1 `, Since `stratovolcano` occurs in one doc.
   
#### iii. TfiDF
    
    TfiDF = TF * iDF
    
    Therefore,
        `TfiDF("is") = 1000 * 0 = 0`
        `TfiDF("Everest") = 50 * 0.6 = 30`
        `TfiDF("stratovolcano") = 2 * 1 = 2`
    
   

Let's calcuate TfiDF for our sample train data above.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
sample_train_tfidf = tfidf_vec.fit_transform(sample_train_data)
```


```python
pd.DataFrame(sample_train_tfidf.toarray(), columns=tfidf_vec.get_feature_names())
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline</th>
      <th>and</th>
      <th>assembly</th>
      <th>country</th>
      <th>delays</th>
      <th>dispute</th>
      <th>encourage</th>
      <th>entrepreneurs</th>
      <th>formation</th>
      <th>from</th>
      <th>...</th>
      <th>go</th>
      <th>is</th>
      <th>looking</th>
      <th>national</th>
      <th>process</th>
      <th>startup</th>
      <th>surcharges</th>
      <th>to</th>
      <th>tuesday</th>
      <th>up</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.423394</td>
      <td>0.000000</td>
      <td>0.423394</td>
      <td>0.423394</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.423394</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.423394</td>
      <td>0.322002</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.350139</td>
      <td>0.000000</td>
      <td>0.350139</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.350139</td>
      <td>0.350139</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.350139</td>
      <td>0.350139</td>
      <td>0.000000</td>
      <td>0.266290</td>
      <td>0.350139</td>
      <td>0.000000</td>
      <td>0.266290</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.363255</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.363255</td>
      <td>...</td>
      <td>0.363255</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.363255</td>
      <td>0.276265</td>
      <td>0.363255</td>
      <td>0.363255</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>



Let us discuss few questions regarding feature engineering.

#### Why do you need to vectorize the words?
 - Raw text data can not be directly fed into machine learning model. So, text to vector conversion PIPE-line consists of following process:

    * Tokenization : When we convert our text into list of words by removing stop words and punctuations, we assign individual string an id value.
     
    * Frequency counting :  We count the occurence of individual token into each of the document.
    * Normalizing :  We reduce the importance of token having frequent occurence in majority of documents by giving it lower weight.

  At the end, the token occurrene frequency is our feature for the model.reated as a feature.
  
####  Why do you need to do data standardization?
   - 

### C. Choose estimator

I will be using `SVM` as my estimator. We will not talk details on SVM because it might take long hours to understand the algorithms.
In brief, `linear SVM` tries to find an optimal vector ( a line) sperating two data points i.e classifying data. 



```python
# news_classifier.py
import pickle
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

class NewsClassification:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        self.train_vectors = None
        self.test_vectors = None
        self.vectorizer = TfidfVectorizer()
        self.model = None

    def prepare_feature_vectors(self):
        # Train the feature vectors
        self.train_vectors = self.vectorizer.fit_transform(self.train_data)
        # Apply model on test data
        # : since they have already been fit to the training set
        self.test_vectors = self.vectorizer.transform(self.test_data)

    def prepare_model(self):
        self.model = svm.SVC(kernel='linear', class_weight="balanced")
        self.model.fit(self.train_vectors, self.train_labels)


if __name__ == '__main__':
    news_classifier = NewsClassification()
    data = pd.read_csv(
        'news_corpus/data.csv',
        names=['text', 'label'], dtype={'label': object})

    data = data[data['label'].map(len) <= 2]
    train, test = train_test_split(data, test_size=0.2)

    news_classifier = NewsClassification()
    news_classifier.train_data, news_classifier.train_labels =\
        train['text'], train['label']
    news_classifier.test_data, news_classifier.test_labels =\
        test['text'], test['label']

    news_classifier.prepare_feature_vectors()
    news_classifier.prepare_model()
    
```

#### D. Cross Validation

Cross Validation is a tool to assess the predictive performance of our model. During training, we only fit our train data and look at the model performance of model in our in-sample data alone. Using cross validation, we are trying to observe the model behaviour in new data in terms of accuracy. There are different approaches for computing cross validation.
Holding some subset of data from our training samples and finally using this set to evaluate the model performace is one of the approach. This is what we did above using train_test_split.
But, this approach is not optimal. Here, only a certain portion of data contribute for model development. This can be migiated by using cross-validation - running sequence of fits on the model using subsets which act as both training and validation set. We used cross_val_score to achieve this. cross_val_score uses StratifiedKFold to creates k folds of data then compute accuracy in individual run.

The very basic approach is  k-fold cross validation.Here, the training set is split into k folds. And, for each of the k folds:
    1. Model training is done in k-1 of the folds as training data
    2. The trained model is validated against remaining part of the data.

Cross validation can be used for different purposes:
        1. Cross Validation for model selection
        2. Cross Validation for Paramters tuning (Hyperparameters)
        3. Cross Validation for feature selection
        


#### D.1 Cross Validation for model selection
We would like to see whether Naive Bayes classifier or SVM is suitable for our data.


```python
from sklearn.model_selection import cross_val_score
# cross validation score for Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
print(cross_val_score(nb, news_classifier.train_vectors,
        news_classifier.train_labels, cv=10, scoring='accuracy').mean())

```

    0.71135591129



```python
# cross validation score for SVM
svm_model = svm.SVC(kernel='linear')
print(cross_val_score(svm_model, news_classifier.train_vectors,
        news_classifier.train_labels,cv=10, scoring='accuracy').mean())
```

    0.73772542533


##### We saw that SVM is performing better than Naive Bayes classifier

#### D.2 Cross Validation for Paramters tuning

Until now, we are looking at training data to measure the accuracy of our model. But, there are also other parameters which are not the part of model training process. These parameters are called hyperparameters, which defines areas like - the complexity of model and how fast the model should learn. So, Hyperparameters can be defined as knobs or levels we fed for model building. 
On the other hand, model parameters are the properties that are learnt during training by the model. For example, if we are looking for Natural Lanuage processing model: word frequency, sentence length or n-grams per word are Model parameters.
During model development, we try to optimize model parameters looking into the performance of some loss function whereas optimizing hyperpameters is handled by investigatin various settings of the parameter values and located only after we observe the model accuracy. 
In this process, we will observe variations of SVM hyper-parameters "kernel”, "gamma" and "C" in order to get best model performing values.


GridSearchCV splits your given data input into Train and CV set and train algorithm with train set searching for the best hyperparameters using the CV set.


```python
import numpy as np
from sklearn.grid_search import GridSearchCV


param_grid = [
  {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
  {'C': [0.01, 0.1, 1, 10, 100], 'gamma': np.logspace(-3, 2, 20), 'kernel': ['rbf']},
 ]

clf = GridSearchCV(svm.SVC(), param_grid,  cv=10,)
clf = clf.fit(news_classifier.train_vectors, news_classifier.train_labels)
print("Best estimator found by grid search:")
print(clf.best_estimator_)
```


```python
# Best parameters for the model found using grid search
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
```

    Best C: 1
    Best Kernel: linear
    Best Gamma: auto




#### E. Prediction with test data


```python
# let's run prediction with our test data
# Predict new data with model loaded from disk
prediction = clf.best_estimator_.predict(news_classifier.test_vectors)
# prediction = news_classifier.model.predict(news_classifier.test_vectors)

test['prediction'] = prediction

test.tail(5)
```

    /home/leapfrong/workspace/envs/envpractice/lib/python3.5/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      after removing the cwd from sys.path.





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1792</th>
      <td>Honda splits Acura into its own division to revitalize brand</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12656</th>
      <td>Low Quality.</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9941</th>
      <td>Titanfall on Xbox One: Frame-Rate Is Uneven, Struggles A Lot Even At 792p  ...</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6939</th>
      <td>Colorado made $3.5m 'pot' tax in January</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11696</th>
      <td>Great pork sandwich.</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>






```python

```

### Analysis

#### Model Evaluation

There are different appoaches evaluating for model performance and error analysis. And, this particularly depends upon type of problem we are solving. Since our problem falls under classification, we need to identify which class defines positive result and which one negative. In our case, we assume that predicting 'positive' class is a positive result. 
If our model is predicting input text as 'positive' and it really has 'positive' label in test data, then it is called True positive. Why True Positive? 'True' because the real class of the input data matched the prediction, and 'Positive' as we have assumed predicting 'positive' is Postive result above.
On the other hand, if our model predicted an input text as 'positive' though it is 'negative' in real, it is called False Positive. 'False' because the real class of the input data did not match with the predicted value, and 'Positive' as prediction 'positve' class matched our assumption above.

While deriving model performance and error analysis, we should always consider test data. Using training data will not make much sense as our model will be baised to give the actual prediction.


##### i. Accuracy
  - It measures percentage of correct predictions .i.e the number of correct predictions
 made by our model divided by the total number of predictions.


```python
print(clf.best_estimator_.score(
        news_classifier.test_vectors, news_classifier.test_labels))

```

    0.731187122736



```python
# let's also look at the best score for training data derived from cross validation

print(clf.best_estimator_.best_score_)
```

    0.899859098229





```python

```

####  NOTE : Test data accuracy is near to cross validation accuracy.

##### ii. Confusion Matrix
  - This matrix helps you to understand the types of errors made by our classifier.


```python
conf_mat = metrics.confusion_matrix(
    news_classifier.test_labels, prediction,
    labels=['1', '-1', '0']
)
print(conf_mat)
```

    [[168  56  39]
     [ 35 139  31]
     [ 22  32 129]]


* Here, we have 222 (128+94) misclassified data out of 1000.


```python
# Let's plot the matrix so that it will be more easier to see the model performance
import matplotlib.pyplot as plt
import numpy as np

plt.clf()
img = plt.imshow(conf_mat, interpolation='nearest')
img.set_cmap('winter_r')
class_names = ['Positive', 'Negative', 'Neutral']
plt.title('Positive : Negative Sentiment Confusion Matrix', fontsize="15", color="red")
plt.ylabel('True label')
plt.xlabel('Predicted label')

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
conf_labels = [['TP', 'FN'], ['FP', 'TN']]
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(conf_mat[i][j]))
plt.show()

```


![png](sentiment_analysis_files/sentiment_analysis_58_0.png)



TP = True positive , FN = False Negative, 
 - TP_A (168) : sentiments our model thinks are "positive" and are "positive" in reality.
 - FN_A (56 + 39 = 95) : sentiments our model thinks are "negative" or "neutral" but they are "positive".
 - FP_A (35 + 22 = 57) : sentiments our model thinks are "positive" but they are "negative" or "neutral" in reality.
 - TP_B (139) : sentiments our model thinks are "negative" and are "negative" in reality.
 - FN_B (35 + 31 = 66) : sentiments our model thinks are "positive" or "neutral" but they are "negative" in reality.
 - FP_B (56 + 32 = 88) : sentiments our model thinks are "negative" but they are "positive" or "neutral" in reality.

#### Recall
##### -  There are total 217 (122 + 95) positive labels. And, our model is successfully identified 168 as positive
 i.e recall = 168/217 = 0.774  (TP/(TP+FN))
  -  Recall measures the ability of the classifier to find all the positive data points.
  
#### Precision
##### -  Out of 225 (168 + 57) predicted positive labels, only 168 are real positives.
 i.e precision = 168 /225 = 0.746 (TP/(TP+FP))
  -  It simply measures the proportion of correct positive predictions out of all positive predictions made by our model.

iii. Overfitting and Underfitting:

In order to identify the poorness of our model, observing model fit is crucial. The very basic approach to determine overfitting or underfitting of our model is by plotting prediction error on the training data and test data.


If our model performs better on the training data and does poorly on test set, it means our model is memorizing the training data pattern but unable to fit or genaralize unseen data. This phenonemon of perfect fit on training data is called overfitting. This is the case when our model too complex. For example, if we are using polynomial model of degree 6, it will try to fit your training data perfectly. It seems polynomial model is a good choice for your actual data distribution, but it may try to remember every data points rather than capturing true pattern. In this scenario, if you add extra points (more training data), your model will be extra flattened. In other word, it will follow the data movement rather than data pattern. If you plot prediction error on training and test data, and the curves are far from each other then it is overfitting.

The very basic approach would be plotting learning curve to see the behaviour of model with increasing training sample size. If we see that the cap between traiining error and validation error is about to close with inreasing training size, we need to collect more training data.
Another solution would be looking for simpler classifier. If you have small dataset, choosing simple model always prevent overfitting. 

On the other hand, there might be a case where our model don't even capture the patterns from training data. That means even performing poorly in training set. This is called underfitting. In the plot, if the curves are close enough, it shows symptoms of underfitting. This can be minimized by using more complex model / using more features for model development.


iv.Validation Curve

Validation curve is plotted in order to observe the model performance in terms of model complexity. Here, we try to vary a parameter which contributes to control model complexity (in our case - 'Gamma') and calculates the error on both training and test data.



```python
x = news_classifier.train_vectors
y = news_classifier.train_labels
# Create a gamma values from 10^-6 to 10^0.5 with 20 equally spaced intervals
param_range = np.logspace(-6, 0.5, 20)
train_scores, validation_scores = validation_curve(
    svm.SVC(kernel='rbf'), x, y,
    param_name='gamma',
    param_range=param_range, cv=10)

# Calculate mean for training and test set scores
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(validation_scores, axis=1)


# Plot the mean train score and validation score
plt.plot(param_range, test_scores_mean, label='cross-validation')
plt.plot(param_range, train_scores_mean, label='training')

# Create plot
plt.title("Validation Curve With SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

```

<img src="validation_curve.png">

In the plot above:
1. The training score is improving with increased model complexity and is higher than validation score. This means our model is performing better with training set but not with test data.
2. On the left side of the curve (lower gamma values - lower model complexity - a high-bias), both training and validation scores are minimal. This means model is not performing good for both the training and test set. This is called underfitting.

3. Right side of the curve ( higher model complexity - a high-variance) shows over-fit. This means the model can perform better with training data but fails to predict unseen data.

4. For an intermediate value of Gamma, both scores have maximum value. This level of model complexity shows the optimal trade-off between bias and variance.






```python

```

v. Learning curve
 - Learning curve depicts the behavior of model while increasing the number of train data points.
 http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


```python
title = 'Learning Curve : (SVM, linear kernel, $\gamma=%.4f$)' % clf.best_estimator_.gamma
estimator = svm.SVC(kernel='rbf', gamma=clf.best_estimator_.gamma, C = 10)
plot_learning_curve(
    clf.best_estimator_, title, news_classifier.train_vectors,
    news_classifier.train_labels,
    cv=10)
```

<img src="learningcurve.png">

 - In the figure, we can see that training score attends maximum value regardless of increasing size of training data. 
 - Also, validation score increases with growing training instances
 - So, it can be concluded that 'rbf' kernel is a high variance estimator which over-fits our data. This high variance model can be improved by adding more training samples.


In my opinion, sentiment analysis can not be totally derived from concrete science as we are dealing with emotions and feelings of people. The best observation could be testing whether our classifier is good to predict correct output in unseen data set. And, at the same time, the selection of best classifier depends upon your need as well. For example, you are developing a model to classify patients cancer diagnosis report. Here, if you model happens to have higher False Positive value (means you model is saying that patient have cancer when they do not), it is a critical issue. In such case, you should look for a better model, try to use different classifier with different parameters for your test data, and figure out the best one.s


```python

```
