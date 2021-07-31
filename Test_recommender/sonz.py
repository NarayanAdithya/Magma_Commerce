{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reading data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package averaged_perceptron_tagger to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package averaged_perceptron_tagger is already up-to-\n",
      "[nltk_data]       date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Unzipping corpora\\stopwords.zip.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')\n",
    "nltk.download('averaged_perceptron_tagger')\n",
    "nltk.download('wordnet')\n",
    "nltk.download('stopwords')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('articles_bbc_2018_01_30.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(309, 2)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data.dropna().reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(308, 2)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Keeping English articles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-8-ee4739df7ffd>:3: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0\n",
      "Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`\n",
      "  tqdm_notebook().pandas()\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "30bc92b13ff14690aac827505daece5b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0it [00:00, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from langdetect import detect\n",
    "from tqdm import tqdm_notebook\n",
    "tqdm_notebook().pandas()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ce9d0eedb5424ac89171828dd1f2b888",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/308 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['lang'] = data.articles.progress_map(detect)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "en    257\n",
       "fa      9\n",
       "fr      7\n",
       "id      5\n",
       "uk      4\n",
       "ar      4\n",
       "ru      4\n",
       "hi      4\n",
       "vi      4\n",
       "sw      3\n",
       "es      2\n",
       "pt      2\n",
       "tr      2\n",
       "de      1\n",
       "Name: lang, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.lang.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data.loc[data.lang=='en']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Tokenization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.tokenize import sent_tokenize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c25f443bf56144bc93f35939944b032e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/257 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data['sentences'] = data.articles.progress_map(sent_tokenize)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.tokenize import word_tokenize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ae478218ef2445a3a8eafb2444df53ec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/257 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[['Image', 'copyright', 'PA/EPA', 'Image', 'caption', 'Oligarch', 'Roman', 'Abramovich', '(', 'l', ')', 'and', 'PM', 'Dmitry', 'Medvedev', 'are', 'on', 'the', 'list', 'Russian', 'President', 'Vladimir', 'Putin', 'says', 'a', 'list', 'of', 'officials', 'and', 'businessmen', 'close', 'to', 'the', 'Kremlin', 'published', 'by', 'the', 'US', 'has', 'in', 'effect', 'targeted', 'all', 'Russian', 'people', '.'], ['The', 'list', 'names', '210', 'top', 'Russians', 'as', 'part', 'of', 'a', 'sanctions', 'law', 'aimed', 'at', 'punishing', 'Moscow', 'for', 'meddling', 'in', 'the', 'US', 'election', '.'], ['However', ',', 'the', 'US', 'stressed', 'those', 'named', 'were', 'not', 'subject', 'to', 'new', 'sanctions', '.']]\n"
     ]
    }
   ],
   "source": [
    "data['tokens_sentences'] = data['sentences'].progress_map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])\n",
    "print(data['tokens_sentences'].head(1).tolist()[0][:3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Lemmatizing with POS tagging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import pos_tag"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fd760c745d2049cba77ea598e521b1aa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/257 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[('Image', 'NN'), ('copyright', 'NN'), ('PA/EPA', 'NNP'), ('Image', 'NNP'), ('caption', 'NN'), ('Oligarch', 'NNP'), ('Roman', 'NNP'), ('Abramovich', 'NNP'), ('(', '('), ('l', 'NN'), (')', ')'), ('and', 'CC'), ('PM', 'NNP'), ('Dmitry', 'NNP'), ('Medvedev', 'NNP'), ('are', 'VBP'), ('on', 'IN'), ('the', 'DT'), ('list', 'NN'), ('Russian', 'NNP'), ('President', 'NNP'), ('Vladimir', 'NNP'), ('Putin', 'NNP'), ('says', 'VBZ'), ('a', 'DT'), ('list', 'NN'), ('of', 'IN'), ('officials', 'NNS'), ('and', 'CC'), ('businessmen', 'NNS'), ('close', 'RB'), ('to', 'TO'), ('the', 'DT'), ('Kremlin', 'NNP'), ('published', 'VBN'), ('by', 'IN'), ('the', 'DT'), ('US', 'NNP'), ('has', 'VBZ'), ('in', 'IN'), ('effect', 'NN'), ('targeted', 'VBN'), ('all', 'DT'), ('Russian', 'JJ'), ('people', 'NNS'), ('.', '.')], [('The', 'DT'), ('list', 'NN'), ('names', 'RB'), ('210', 'CD'), ('top', 'JJ'), ('Russians', 'NNPS'), ('as', 'IN'), ('part', 'NN'), ('of', 'IN'), ('a', 'DT'), ('sanctions', 'NNS'), ('law', 'NN'), ('aimed', 'VBN'), ('at', 'IN'), ('punishing', 'VBG'), ('Moscow', 'NNP'), ('for', 'IN'), ('meddling', 'VBG'), ('in', 'IN'), ('the', 'DT'), ('US', 'NNP'), ('election', 'NN'), ('.', '.')], [('However', 'RB'), (',', ','), ('the', 'DT'), ('US', 'NNP'), ('stressed', 'VBD'), ('those', 'DT'), ('named', 'VBN'), ('were', 'VBD'), ('not', 'RB'), ('subject', 'JJ'), ('to', 'TO'), ('new', 'JJ'), ('sanctions', 'NNS'), ('.', '.')]]\n"
     ]
    }
   ],
   "source": [
    "data['POS_tokens'] = data['tokens_sentences'].progress_map(lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])\n",
    "print(data['POS_tokens'].head(1).tolist()[0][:3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Inspired from https://stackoverflow.com/a/15590384\n",
    "from nltk.corpus import wordnet\n",
    "\n",
    "def get_wordnet_pos(treebank_tag):\n",
    "\n",
    "    if treebank_tag.startswith('J'):\n",
    "        return wordnet.ADJ\n",
    "    elif treebank_tag.startswith('V'):\n",
    "        return wordnet.VERB\n",
    "    elif treebank_tag.startswith('N'):\n",
    "        return wordnet.NOUN\n",
    "    elif treebank_tag.startswith('R'):\n",
    "        return wordnet.ADV\n",
    "    else:\n",
    "        return ''\n",
    "\n",
    "from nltk.stem.wordnet import WordNetLemmatizer\n",
    "lemmatizer = WordNetLemmatizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c404163f54eb40b7819bcea98922b811",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/257 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Lemmatizing each word with its POS tag, in each sentence\n",
    "data['tokens_sentences_lemmatized'] = data['POS_tokens'].progress_map(\n",
    "    lambda list_tokens_POS: [\n",
    "        [\n",
    "            lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) \n",
    "            if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS\n",
    "        ] \n",
    "        for tokens_POS in list_tokens_POS\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[['Image',\n",
       "  'copyright',\n",
       "  'PA/EPA',\n",
       "  'Image',\n",
       "  'caption',\n",
       "  'Oligarch',\n",
       "  'Roman',\n",
       "  'Abramovich',\n",
       "  '(',\n",
       "  'l',\n",
       "  ')',\n",
       "  'and',\n",
       "  'PM',\n",
       "  'Dmitry',\n",
       "  'Medvedev',\n",
       "  'be',\n",
       "  'on',\n",
       "  'the',\n",
       "  'list',\n",
       "  'Russian',\n",
       "  'President',\n",
       "  'Vladimir',\n",
       "  'Putin',\n",
       "  'say',\n",
       "  'a',\n",
       "  'list',\n",
       "  'of',\n",
       "  'official',\n",
       "  'and',\n",
       "  'businessmen',\n",
       "  'close',\n",
       "  'to',\n",
       "  'the',\n",
       "  'Kremlin',\n",
       "  'publish',\n",
       "  'by',\n",
       "  'the',\n",
       "  'US',\n",
       "  'have',\n",
       "  'in',\n",
       "  'effect',\n",
       "  'target',\n",
       "  'all',\n",
       "  'Russian',\n",
       "  'people',\n",
       "  '.'],\n",
       " ['The',\n",
       "  'list',\n",
       "  'names',\n",
       "  '210',\n",
       "  'top',\n",
       "  'Russians',\n",
       "  'as',\n",
       "  'part',\n",
       "  'of',\n",
       "  'a',\n",
       "  'sanction',\n",
       "  'law',\n",
       "  'aim',\n",
       "  'at',\n",
       "  'punish',\n",
       "  'Moscow',\n",
       "  'for',\n",
       "  'meddle',\n",
       "  'in',\n",
       "  'the',\n",
       "  'US',\n",
       "  'election',\n",
       "  '.'],\n",
       " ['However',\n",
       "  ',',\n",
       "  'the',\n",
       "  'US',\n",
       "  'stress',\n",
       "  'those',\n",
       "  'name',\n",
       "  'be',\n",
       "  'not',\n",
       "  'subject',\n",
       "  'to',\n",
       "  'new',\n",
       "  'sanction',\n",
       "  '.']]"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['tokens_sentences_lemmatized'].head(1).tolist()[0][:3]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Regrouping tokens and removing stop words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']\n",
    "stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']\n",
    "my_stopwords = stopwords.words('English') + stopwords_verbs + stopwords_other"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from itertools import chain # to flatten list of sentences of tokens into list of tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))\n",
    "data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() \n",
    "                                                    and token.lower() not in my_stopwords and len(token)>1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['oligarch',\n",
       " 'roman',\n",
       " 'abramovich',\n",
       " 'pm',\n",
       " 'dmitry',\n",
       " 'medvedev',\n",
       " 'list',\n",
       " 'russian',\n",
       " 'president',\n",
       " 'vladimir',\n",
       " 'putin',\n",
       " 'list',\n",
       " 'official',\n",
       " 'businessmen',\n",
       " 'close',\n",
       " 'kremlin',\n",
       " 'publish',\n",
       " 'us',\n",
       " 'effect',\n",
       " 'target',\n",
       " 'russian',\n",
       " 'people',\n",
       " 'list',\n",
       " 'names',\n",
       " 'top',\n",
       " 'russians',\n",
       " 'part',\n",
       " 'sanction',\n",
       " 'law',\n",
       " 'aim']"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['tokens'].head(1).tolist()[0][:30]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LDA"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Prepare bi-grams and tri-grams"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "from gensim.models import Phrases"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'tokens'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3079\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3080\u001b[1;33m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3081\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\hashtable_class_helper.pxi\u001b[0m in \u001b[0;36mpandas._libs.hashtable.PyObjectHashTable.get_item\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'tokens'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-19-9465f19fabc9>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtokens\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdata\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'tokens'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtolist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mbigram_model\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mPhrases\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtokens\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mtrigram_model\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mPhrases\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbigram_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtokens\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmin_count\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mtokens\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrigram_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mbigram_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtokens\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3022\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[1;33m>\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3023\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3024\u001b[1;33m             \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3025\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3026\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\pandas\\core\\indexes\\base.py\u001b[0m in \u001b[0;36mget_loc\u001b[1;34m(self, key, method, tolerance)\u001b[0m\n\u001b[0;32m   3080\u001b[0m                 \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3081\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 3082\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0merr\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   3083\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3084\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mtolerance\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyError\u001b[0m: 'tokens'"
     ]
    }
   ],
   "source": [
    "tokens = data['tokens'].tolist()\n",
    "bigram_model = Phrases(tokens)\n",
    "trigram_model = Phrases(bigram_model[tokens], min_count=1)\n",
    "tokens = list(trigram_model[bigram_model[tokens]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Prepare objects for LDA gensim implementation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "from gensim import corpora"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tokens' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-40-06f3b7cb86bf>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mdictionary_LDA\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcorpora\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDictionary\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtokens\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mdictionary_LDA\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfilter_extremes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mno_below\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mcorpus\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mdictionary_LDA\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdoc2bow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtok\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mtok\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtokens\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'tokens' is not defined"
     ]
    }
   ],
   "source": [
    "dictionary_LDA = corpora.Dictionary(tokens)\n",
    "dictionary_LDA.filter_extremes(no_below=3)\n",
    "corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Running LDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from gensim import models\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(123456)\n",
    "num_topics = 20\n",
    "%time lda_model = models.LdaModel(corpus, num_topics=num_topics, \\\n",
    "                                  id2word=dictionary_LDA, \\\n",
    "                                  passes=4, alpha=[0.01]*num_topics, \\\n",
    "                                  eta=[0.01]*len(dictionary_LDA.keys()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Quick exploration of LDA results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Looking at topics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):\n",
    "    print(str(i)+\": \"+ topic)\n",
    "    print()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Allocating topics to documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Image copyright PA/EPA Image caption Oligarch Roman Abramovich (l) and PM Dmitry Medvedev are on the list\r\n",
      "\r\n",
      "Russian President Vladimir Putin says a list of officials and businessmen close to the Kremlin published by the US has in effect targeted all Russian people.\r\n",
      "\r\n",
      "The list names 210 top Russians as part of a sanctions law aimed at punishing Moscow for meddling in the US election.\r\n",
      "\r\n",
      "However, the US stressed those named were not subject to new sanctions.\r\n",
      "\r\n",
      "Mr Putin said the list was an unfr\n"
     ]
    }
   ],
   "source": [
    "print(data.articles.loc[0][:500])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'lda_model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-23-828908f2c471>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mlda_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'lda_model' is not defined"
     ]
    }
   ],
   "source": [
    "lda_model[corpus[0]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Predicting topics on unseen documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'word_tokenize' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-24-31336e0f5a5e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m Here, The New York Times deconstructs how Mr. Tucker’s now-deleted declaration on Twitter the night after the election turned into a fake-news phenomenon. It is an example of how, in an ever-connected world where speed often takes precedence over truth, an observation by a private citizen can quickly become a talking point, even as it is being proved false.'''\n\u001b[1;32m---> 10\u001b[1;33m \u001b[0mtokens\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mword_tokenize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdocument\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     11\u001b[0m \u001b[0mtopics\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlda_model\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshow_topics\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mformatted\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_topics\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mnum_topics\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_words\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mel\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mround\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mel\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtopics\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mel\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mel\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mlda_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mdictionary_LDA\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdoc2bow\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtokens\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcolumns\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'topic #'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'weight'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'words in topic'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'word_tokenize' is not defined"
     ]
    }
   ],
   "source": [
    "document = '''Eric Tucker, a 35-year-old co-founder of a marketing company in Austin, Tex., had just about 40 Twitter followers. But his recent tweet about paid protesters being bused to demonstrations against President-elect Donald J. Trump fueled a nationwide conspiracy theory — one that Mr. Trump joined in promoting. \n",
    "\n",
    "Mr. Tucker's post was shared at least 16,000 times on Twitter and more than 350,000 times on Facebook. The problem is that Mr. Tucker got it wrong. There were no such buses packed with paid protesters.\n",
    "\n",
    "But that didn't matter.\n",
    "\n",
    "While some fake news is produced purposefully by teenagers in the Balkans or entrepreneurs in the United States seeking to make money from advertising, false information can also arise from misinformed social media posts by regular people that are seized on and spread through a hyperpartisan blogosphere.\n",
    "\n",
    "Here, The New York Times deconstructs how Mr. Tucker’s now-deleted declaration on Twitter the night after the election turned into a fake-news phenomenon. It is an example of how, in an ever-connected world where speed often takes precedence over truth, an observation by a private citizen can quickly become a talking point, even as it is being proved false.'''\n",
    "tokens = word_tokenize(document)\n",
    "topics = lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20)\n",
    "pd.DataFrame([(el[0], round(el[1],2), topics[el[0]][1]) for el in lda_model[dictionary_LDA.doc2bow(tokens)]], columns=['topic #', 'weight', 'words in topic'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Advanced exploration of LDA results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Allocation of topics in all documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'lda_model' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-25-7776a9eb2c74>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtopics\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mlda_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-25-7776a9eb2c74>\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mtopics\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mlda_model\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mcorpus\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'lda_model' is not defined"
     ]
    }
   ],
   "source": [
    "topics = [lda_model[corpus[i]] for i in range(len(data))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "      <th>13</th>\n",
       "      <th>14</th>\n",
       "      <th>15</th>\n",
       "      <th>16</th>\n",
       "      <th>17</th>\n",
       "      <th>18</th>\n",
       "      <th>19</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.038537</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.091301</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.869287</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    0    1    2    3    4    5    6    7    8         9    10   11   12   13  \\\n",
       "0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  0.038537  NaN  NaN  NaN  NaN   \n",
       "\n",
       "    14        15   16   17        18   19  \n",
       "0  NaN  0.091301  NaN  NaN  0.869287  NaN  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def topics_document_to_dataframe(topics_document, num_topics):\n",
    "    res = pd.DataFrame(columns=range(num_topics))\n",
    "    for topic_weight in topics_document:\n",
    "        res.loc[0, topic_weight[0]] = topic_weight[1]\n",
    "    return res\n",
    "\n",
    "topics_document_to_dataframe([(9, 0.03853655432967504), (15, 0.09130117862212643), (18, 0.8692868808484044)], 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'topics' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-27-ce1083c907df>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m# Like TF-IDF, create a matrix of topic weighting, with documents as rows and topics as columns\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mdocument_topic\u001b[0m \u001b[1;33m=\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtopics_document_to_dataframe\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtopics_document\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_topics\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mnum_topics\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mtopics_document\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mtopics\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m   \u001b[1;33m.\u001b[0m\u001b[0mreset_index\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdrop\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfillna\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'topics' is not defined"
     ]
    }
   ],
   "source": [
    "# Like TF-IDF, create a matrix of topic weighting, with documents as rows and topics as columns\n",
    "document_topic = \\\n",
    "pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]) \\\n",
    "  .reset_index(drop=True).fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "document_topic.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "# Which document are about topic 14\n",
    "document_topic.sort_values(14, ascending=False)[14].head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "print(data.articles.loc[91][:1000])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Looking at the distribution of topics in all documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'document_topic' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-28-39e9e9c304f9>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mget_ipython\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrun_line_magic\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'matplotlib'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'inline'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mseaborn\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m;\u001b[0m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrc\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m{\u001b[0m\u001b[1;34m'figure.figsize'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m10\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m20\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0msns\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mheatmap\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdocument_topic\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mdocument_topic\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0midxmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msort_values\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mindex\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'document_topic' is not defined"
     ]
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "import seaborn as sns; sns.set(rc={'figure.figsize':(10,20)})\n",
    "sns.heatmap(document_topic.loc[document_topic.idxmax(axis=1).sort_values().index])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set(rc={'figure.figsize':(10,5)})\n",
    "document_topic.idxmax(axis=1).value_counts().plot.bar(color='lightblue')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualizing topics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf\n",
    "# Here a short legend to explain the vis:\n",
    "# size of bubble: proportional to the proportions of the topics across the N total tokens in the corpus\n",
    "# red bars: estimated number of times a given term was generated by a given topic\n",
    "# blue bars: overall frequency of each term in the corpus\n",
    "# -- Relevance of words is computed with a parameter lambda\n",
    "# -- Lambda optimal value ~0.6 (https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)\n",
    "%matplotlib inline\n",
    "import pyLDAvis\n",
    "import pyLDAvis.gensim\n",
    "vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)\n",
    "pyLDAvis.enable_notebook()\n",
    "pyLDAvis.display(vis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package averaged_perceptron_tagger to\n",
      "[nltk_data]     C:\\Users\\ASUS\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Unzipping taggers\\averaged_perceptron_tagger.zip.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')\n",
    "nltk.download('averaged_perceptron_tagger')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  },
  "widgets": {
   "state": {
    "429f605e71e34a868915456ee4fb20b8": {
     "views": [
      {
       "cell_index": 14
      }
     ]
    },
    "46817b7b86ca4b65a4bd631d2e7d777f": {
     "views": [
      {
       "cell_index": 21
      }
     ]
    },
    "8559963a3ac94b19a5b3ea6214032316": {
     "views": [
      {
       "cell_index": 19
      }
     ]
    },
    "96323249f1884bd5b62f87c3145932df": {
     "views": [
      {
       "cell_index": 8
      }
     ]
    },
    "d9d5e9ab22a947f398aad3ed9c365fcf": {
     "views": [
      {
       "cell_index": 16
      }
     ]
    },
    "fa80f472c5f14bcd9a7ba0befad06400": {
     "views": [
      {
       "cell_index": 9
      }
     ]
    }
   },
   "version": "1.2.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
