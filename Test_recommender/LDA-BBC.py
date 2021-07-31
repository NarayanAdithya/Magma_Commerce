

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv('articles_bbc_2018_01_30.csv')


# In[4]:


data.shape


# In[5]:


data = data.dropna().reset_index(drop=True)


# In[6]:


data.shape


# # Cleaning

# #### Keeping English articles

# In[7]:


from langdetect import detect
from tqdm import tqdmnotebook.tqdm
tqdm_notebook().pandas()


# In[8]:


data['lang'] = data.articles.progress_map(detect)


# In[9]:


data.lang.value_counts()


# In[10]:


data = data.loc[data.lang=='en']


# #### Tokenization

# In[11]:


from nltk.tokenize import sent_tokenize


# In[12]:


data['sentences'] = data.articles.progress_map(sent_tokenize)
data['sentences'].head(1).tolist()[0][:3] # Print the first 3 sentences of the 1st article


# In[13]:


from nltk.tokenize import word_tokenize


# In[14]:


data['tokens_sentences'] = data['sentences'].progress_map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
print(data['tokens_sentences'].head(1).tolist()[0][:3])


# #### Lemmatizing with POS tagging

# In[15]:


from nltk import pos_tag


# In[16]:


data['POS_tokens'] = data['tokens_sentences'].progress_map(lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])
print(data['POS_tokens'].head(1).tolist()[0][:3])


# In[17]:


# Inspired from https://stackoverflow.com/a/15590384
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[18]:


# Lemmatizing each word with its POS tag, in each sentence
data['tokens_sentences_lemmatized'] = data['POS_tokens'].progress_map(
    lambda list_tokens_POS: [
        [
            lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
            if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
        ] 
        for tokens_POS in list_tokens_POS
    ]
)


# In[19]:


data['tokens_sentences_lemmatized'].head(1).tolist()[0][:3]


# #### Regrouping tokens and removing stop words

# In[20]:


from nltk.corpus import stopwords
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
my_stopwords = stopwords.words('English') + stopwords_verbs + stopwords_other


# In[21]:


from itertools import chain # to flatten list of sentences of tokens into list of tokens


# In[22]:


data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                    and token.lower() not in my_stopwords and len(token)>1])


# In[23]:


data['tokens'].head(1).tolist()[0][:30]


# # LDA

# ## Data preparation

# #### Prepare bi-grams and tri-grams

# In[24]:


from gensim.models import Phrases


# In[25]:


tokens = data['tokens'].tolist()
bigram_model = Phrases(tokens)
trigram_model = Phrases(bigram_model[tokens], min_count=1)
tokens = list(trigram_model[bigram_model[tokens]])


# #### Prepare objects for LDA gensim implementation

# In[26]:


from gensim import corpora


# In[27]:


dictionary_LDA = corpora.Dictionary(tokens)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]


# ## Running LDA

# In[28]:


from gensim import models
import numpy as np


# In[29]:


np.random.seed(123456)
num_topics = 20
get_ipython().run_line_magic('time', 'lda_model = models.LdaModel(corpus, num_topics=num_topics,                                   id2word=dictionary_LDA,                                   passes=4, alpha=[0.01]*num_topics,                                   eta=[0.01]*len(dictionary_LDA.keys()))')


# ## Quick exploration of LDA results

# #### Looking at topics

# In[271]:


for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
    print(str(i)+": "+ topic)
    print()


# #### Allocating topics to documents

# In[171]:


print(data.articles.loc[0][:500])


# In[172]:


lda_model[corpus[0]]


# #### Predicting topics on unseen documents

# In[44]:


document = '''Eric Tucker, a 35-year-old co-founder of a marketing company in Austin, Tex., had just about 40 Twitter followers. But his recent tweet about paid protesters being bused to demonstrations against President-elect Donald J. Trump fueled a nationwide conspiracy theory — one that Mr. Trump joined in promoting. 

Mr. Tucker's post was shared at least 16,000 times on Twitter and more than 350,000 times on Facebook. The problem is that Mr. Tucker got it wrong. There were no such buses packed with paid protesters.

But that didn't matter.

While some fake news is produced purposefully by teenagers in the Balkans or entrepreneurs in the United States seeking to make money from advertising, false information can also arise from misinformed social media posts by regular people that are seized on and spread through a hyperpartisan blogosphere.

Here, The New York Times deconstructs how Mr. Tucker’s now-deleted declaration on Twitter the night after the election turned into a fake-news phenomenon. It is an example of how, in an ever-connected world where speed often takes precedence over truth, an observation by a private citizen can quickly become a talking point, even as it is being proved false.'''
tokens = word_tokenize(document)
topics = lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20)
pd.DataFrame([(el[0], round(el[1],2), topics[el[0]][1]) for el in lda_model[dictionary_LDA.doc2bow(tokens)]], columns=['topic #', 'weight', 'words in topic'])


# ## Advanced exploration of LDA results

# #### Allocation of topics in all documents

# In[46]:


topics = [lda_model[corpus[i]] for i in range(len(data))]


# In[47]:


def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

topics_document_to_dataframe([(9, 0.03853655432967504), (15, 0.09130117862212643), (18, 0.8692868808484044)], 20)


# In[48]:


# Like TF-IDF, create a matrix of topic weighting, with documents as rows and topics as columns
document_topic = pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics])   .reset_index(drop=True).fillna(0)


# In[49]:


document_topic.head()


# In[50]:


# Which document are about topic 14
document_topic.sort_values(14, ascending=False)[14].head(20)


# In[51]:


print(data.articles.loc[91][:1000])


# #### Looking at the distribution of topics in all documents

# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set(rc={'figure.figsize':(10,20)})
sns.heatmap(document_topic.loc[document_topic.idxmax(axis=1).sort_values().index])


# In[71]:


sns.set(rc={'figure.figsize':(10,5)})
document_topic.idxmax(axis=1).value_counts().plot.bar(color='lightblue')


# #### Visualizing topics

# In[269]:


# https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf
# Here a short legend to explain the vis:
# size of bubble: proportional to the proportions of the topics across the N total tokens in the corpus
# red bars: estimated number of times a given term was generated by a given topic
# blue bars: overall frequency of each term in the corpus
# -- Relevance of words is computed with a parameter lambda
# -- Lambda optimal value ~0.6 (https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)
get_ipython().run_line_magic('matplotlib', 'inline')
import pyLDAvis
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)


# In[ ]:





# In[ ]:




