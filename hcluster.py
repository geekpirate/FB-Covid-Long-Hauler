import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io
import nltk
import csv, os
import unicodedata, re , sys
from textblob import TextBlob
# uncomment below lines when the code is compiled for the first time
# nltk.download('stopwords')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
time = [
  'today', 'day', 'days', 'week', 'night', 'yesterday', 'morning', 'months', 'december', 'second', 'weeks', 'june',
  'january', 'february', 'march', 'april', 'may', 'july', 'august', 'september', 'october', 'november', 'tomorrow',
  'day', 'morning', 'afternoon', 'noon', 'hours'
]
relation =[
  'husband', 'friend', 'mom'
]
stopword = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'ia', 'ita', 'dona', 'symptoms', 'havena', 'got']
stopword += ['again', 'against', 'all', 'almost', 'alone', 'along', 'cana', 'post', 'think', 'couldna', 'group']
stopword += ['already', 'also', 'although', 'always', 'am', 'among', 'work', 'home', 'help', 'thata', 'didna', 'want']
stopword += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'feeling', 'god', 'new', 'bad', 'high', 'loss']
stopword += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'thank', 'thing', 'getting', 'make', 'like']
stopword += ['are', 'around', 'as', 'at', 'back', 'be', 'became', 'family', 'shea', 'taking', 'grade', 'stay']
stopword += ['because', 'become', 'becomes', 'becoming', 'been', 'eat', 'life', 'right', 'come', 'old', 'try']
stopword += ['before', 'beforehand', 'behind', 'being', 'below', 'rate', 'hea', 'going', 'test', 'hospital']
stopword += ['beside', 'besides', 'between', 'beyond', 'bill', 'both', 'doctor', 'nurse', 'say', 'feel', 'issues']
stopword += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'good', 'bless', 'way', 'doesna', 'thought']
stopword += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'term', 'guess', 'trying', 'ray', 'case']
stopword += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due', 'level', 'little', 'stuff', 'program', 'upper']
stopword += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'youa', 'love']
stopword += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'tylenol']
stopword += ['every', 'everyone', 'everything', 'everywhere', 'except']
stopword += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
stopword += ['five', 'for', 'former', 'formerly', 'forty', 'found']
stopword += ['four', 'from', 'front', 'full', 'further', 'get', 'give', 'symptom']
stopword += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'hair']
stopword += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
stopword += ['herself', 'him', 'himself', 'his', 'how', 'however', 'symptoms']
stopword += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
stopword += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopword += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
stopword += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopword += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
stopword += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopword += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
stopword += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopword += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
stopword += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopword += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
stopword += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopword += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
stopword += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopword += ['some', 'somehow', 'someone', 'something', 'sometime']
stopword += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopword += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
stopword += ['then', 'thence', 'there', 'thereafter', 'thereby']
stopword += ['therefore', 'therein', 'thereupon', 'these', 'they']
stopword += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopword += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
stopword += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
stopword += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
stopword += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopword += ['whatever', 'when', 'whence', 'whenever', 'where']
stopword += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopword += ['wherever', 'whether', 'which', 'while', 'whither', 'who', 'aom']
stopword += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'year']
stopword += ['within', 'without', 'would', 'yet', 'you', 'your', 'pray', 'monday', 'tuesday', 'wednesday', 'thursday',
             'friday', 'saturday', 'sunday']
stopword += ['yours', 'yourself', 'yourselves', 'knows', 'know', 'aot', 'type', 'time']


import spacy
import pandas as pd
import nltk
contractions = {
"Id" : "I would",
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
import re, string
import unidecode
from nltk.tokenize import word_tokenize
symptomBag = []
s = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())
newWords = {"burping", "aches", "ache", "coughing"}
words.update(newWords)
#nlp = spacy.load('en_core_web_sm')
s = s.union(set(stopword))
s = s.union(set(relation))
s = s.union(set(time))
allposts = []
wordtree = [['Phrases']]
def patternRecogniton(inputText):
    txt = " ".join(w.lower() for w in nltk.wordpunct_tokenize(inputText) if w.isalpha())
    # w.lower() in words and
    for word in txt.split():
        if word in contractions:
            txt = txt.replace(word, contractions[word])
    txt = re.sub(r'[^\w]', ' ', txt)
    txt = unidecode.unidecode(txt)
    text_tokens = word_tokenize(txt)
    text_tokens = " ".join(word for word in text_tokens if word not in string.punctuation and
                           word not in s and len(word)>2)
    allposts.append(text_tokens)
    #wordtree.append([text_tokens])
    # print(text_tokens)
#def wordTreeProcessing(inputText):
    #txtTree = " ".join(w.lower() for w in nltk.wordpunct_tokenize(inputText) if w.isalpha())
    #txtTree = re.sub(r'[^\w]', ' ', txtTree)
    #txtTree = unidecode.unidecode(txtTree)
    #text_tokensTree = word_tokenize(txtTree)
    #text_tokensTree = " ".join(word for word in text_tokensTree if word not in string.punctuation and len(word)>2)
    #wordtree.append([text_tokensTree])
import xlrd
import xlwt
# load excel with its path
wb = xlrd.open_workbook("C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/1_1 - 1_2 Survivor Corp.xls")
sh = wb.sheet_by_name("1_1 - 1_2 Survivor Corp")
# iterate through excel and display data
for i in range(sh.nrows):
    patternRecogniton(sh.cell_value(i,0))
    #wordTreeProcessing(sh.cell_value(i,0))
#print(wordtree)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=20, max_df=0.4, max_features=20000, ngram_range=(1,1), stop_words='english')
TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
                 encoding='utf-8', input='content',
        lowercase=True, max_df = 0.4, max_features=20000, min_df=20,
        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
        stop_words='english', strip_accents=None, sublinear_tf=False,
        tokenizer=None, use_idf=True,
        vocabulary=None)



# calculate the feature matrix
feature_matrix = vectorizer.fit_transform(allposts)

# display the feature matrix shape
print("features: ",feature_matrix.shape)

'''K means clustering'''

from scipy.cluster.vq import kmeans, vq
import seaborn as sns
def passlabels(label):
    print(label)
    #exit()
    time = [
      'today', 'day', 'days', 'week', 'night', 'yesterday', 'morning', 'months', 'december', 'second', 'weeks', 'june',
      'january', 'february', 'march', 'april', 'may', 'july', 'august', 'september', 'october', 'november', 'tomorrow',
      'day', 'morning', 'afternoon', 'noon', 'hours'
    ]
    relation =[
      'husband', 'friend', 'mom'
    ]
    stopword = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'ia', 'ita', 'dona', 'symptoms', 'havena']
    stopword += ['again', 'against', 'all', 'almost', 'alone', 'along', 'cana', 'post', 'think', 'couldna', 'group']
    stopword += ['already', 'also', 'although', 'always', 'am', 'among', 'work', 'home', 'help', 'thata', 'didna']
    stopword += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'feeling', 'god', 'new', 'bad', 'high', 'loss']
    stopword += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'thank', 'thing', 'getting', 'make']
    stopword += ['are', 'around', 'as', 'at', 'back', 'be', 'became', 'family', 'shea', 'taking', 'grade', 'stay']
    stopword += ['because', 'become', 'becomes', 'becoming', 'been', 'eat', 'life', 'right', 'come', 'old', 'try']
    stopword += ['before', 'beforehand', 'behind', 'being', 'below', 'rate', 'hea', 'going', 'test', 'hospital']
    stopword += ['beside', 'besides', 'between', 'beyond', 'bill', 'both', 'doctor', 'nurse', 'say', 'feel', 'issues']
    stopword += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'good', 'bless', 'way', 'doesna', 'thought']
    stopword += ['co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'term', 'guess', 'trying', 'ray', 'case']
    stopword += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due', 'level', 'little', 'stuff', 'program', 'upper']
    stopword += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'youa', 'love']
    stopword += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'tylenol']
    stopword += ['every', 'everyone', 'everything', 'everywhere', 'except']
    stopword += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
    stopword += ['five', 'for', 'former', 'formerly', 'forty', 'found']
    stopword += ['four', 'from', 'front', 'full', 'further', 'get', 'give', 'symptom']
    stopword += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'hair']
    stopword += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'covid']
    stopword += ['herself', 'him', 'himself', 'his', 'how', 'however', 'symptoms']
    stopword += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
    stopword += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
    stopword += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
    stopword += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
    stopword += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
    stopword += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
    stopword += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
    stopword += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
    stopword += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
    stopword += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
    stopword += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
    stopword += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
    stopword += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
    stopword += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
    stopword += ['some', 'somehow', 'someone', 'something', 'sometime']
    stopword += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
    stopword += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
    stopword += ['then', 'thence', 'there', 'thereafter', 'thereby']
    stopword += ['therefore', 'therein', 'thereupon', 'these', 'they']
    stopword += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
    stopword += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
    stopword += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
    stopword += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
    stopword += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
    stopword += ['whatever', 'when', 'whence', 'whenever', 'where']
    stopword += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
    stopword += ['wherever', 'whether', 'which', 'while', 'whither', 'who', 'aom']
    stopword += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'year']
    stopword += ['within', 'without', 'would', 'yet', 'you', 'your', 'pray', 'monday', 'tuesday', 'wednesday', 'thursday',
                 'friday', 'saturday', 'sunday']
    stopword += ['yours', 'yourself', 'yourselves', 'knows', 'know', 'aot', 'type', 'time']


    import spacy
    contractions = {
    "Id" : "I would",
    "ain't": "am not / are not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
    }
    import pandas as pd
    import nltk
    import re, string
    import unidecode
    from nltk.tokenize import word_tokenize
    symptomBag = []
    s = set(stopwords.words('english'))
    words = set(nltk.corpus.words.words())
    newWords = {"burping", "aches", "ache", "coughing"}
    words.update(newWords)
    #nlp = spacy.load('en_core_web_sm')
    s = s.union(set(stopword))
    s = s.union(set(relation))
    s = s.union(set(time))


    def patternRecogniton(inputText):
      txt = " ".join(w.lower() for w in nltk.wordpunct_tokenize(inputText) if w.isalpha())
      # w.lower() in words and
      for word in txt.split():
          if word in contractions:
              txt = txt.replace(word, contractions[word])
      txt = re.sub(r'[^\w]', ' ', txt)
      txt = unidecode.unidecode(txt)
      text_tokens = word_tokenize(txt)
      text_tokens = [word for word in text_tokens if not word in string.punctuation]
      bigram = [pd.Series(nltk.ngrams(text_tokens, 2))]
      trigram = [pd.Series(nltk.ngrams(text_tokens, 3))]
      for i in range(len(text_tokens)-1):
        # skeptical = ["RB"+"JJ", "CC"+"NN", "JJ"+"NN"+"3", "JJ"+"NN"+"NN", "IN"+"NN"]
        # ignore = ["PRP$"]
        # ignoreWords = ["someone", "please", "thanks"]
        # maybe = ["DT"+"NNS"]
        bipattern = ["NN"+"NN", "NNS"+"NNS", "VBD"+"NN", "VBP"+"NN", "NNP"+"NN", "VBG"+"NNS",  "RB"+"NNS",
                  "NN"+"CC", "VBG"+"NN", "NNP"+"NNP", "VB"+"NNS", "VB"+"NN", "RB"+"VBG", "CC"+"VBG",
                   "NN"+"NNS", "RB"+"VBN", "JJ"+"NN", "DT"+"NNS",
                    "TO"+"VB", "VBP"+"VBD", "VBP"+"VBG","VBG"+"VBG"]
        tripattern = ["NN"+"RB"+"VBZ", "CC"+"NN"+"VBD", "CC"+"CD"+"NN"]
        if ((nltk.pos_tag(bigram[0][i])[0][1])+(nltk.pos_tag(bigram[0][i])[1][1])) in bipattern :
          if i>0:
            #finalWord
            finalWord = nltk.pos_tag(bigram[0][i])[0][0] + " " + nltk.pos_tag(bigram[0][i])[1][0]
            for word in word_tokenize(finalWord):
              if word not in s and len(word)>2:
                symptomBag.append(word)
            # finalWord = " ".join([word for word in word_tokenize(finalWord) if word not in s])
            # if len(finalWord)>1:
              # symptomBag.append(finalWord)
            # if (nltk.pos_tag(bigram[0][i])[0][0] not in s) and (nltk.pos_tag(bigram[0][i])[1][0] not in s):
            #   print(nltk.pos_tag(bigram[0][i])[0][0] + " ," + nltk.pos_tag(bigram[0][i])[1][0])
            #   symptomBag.append(nltk.pos_tag(bigram[0][i])[0][0] + " " + nltk.pos_tag(bigram[0][i])[1][0])
            # elif (nltk.pos_tag(bigram[0][i])[0][0] not in s) and (nltk.pos_tag(bigram[0][i])[1][0] in s):
            #   print(nltk.pos_tag(bigram[0][i])[0][0])
            #   symptomBag.append(nltk.pos_tag(bigram[0][i])[0][0])
            # elif (nltk.pos_tag(bigram[0][i])[0][0] in s) and (nltk.pos_tag(bigram[0][i])[1][0] not in s):
            #   print(nltk.pos_tag(bigram[0][i])[1][0])
            #   symptomBag.append(nltk.pos_tag(bigram[0][i])[1][0])
        if i < len(text_tokens)-3:
          if ((nltk.pos_tag(trigram[0][i])[0][1])+(nltk.pos_tag(trigram[0][i])[1][1])+(nltk.pos_tag(trigram[0][i])[2][1])) in tripattern:
            finalWord = nltk.pos_tag(trigram[0][i])[0][0] + " " + nltk.pos_tag(trigram[0][i])[1][0] + " " + nltk.pos_tag(trigram[0][i])[2][0]
            for word in word_tokenize(finalWord):
              if word not in s and len(word)>2:
                symptomBag.append(word)
            # finalWord = " ".join([word for word in word_tokenize(finalWord) if word not in s])
            # if len(finalWord)>1:
              # symptomBag.append(finalWord)
            #print(nltk.pos_tag(trigram[0][i])[0][0] + " " + nltk.pos_tag(trigram[0][i])[1][0] + " " + nltk.pos_tag(trigram[0][i])[2][0])
            #symptomBag.append(nltk.pos_tag(trigram[0][i])[0][0] + " " + nltk.pos_tag(trigram[0][i])[1][0]
            # + " " + nltk.pos_tag(trigram[0][i])[2][0])

    import xlrd
    import xlwt
    # load excel with its path
    wb = xlrd.open_workbook("C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/1_1 - 1_2 Survivor Corp.xls")
    finalSymptoms = []
    singularSymptoms = []
    texts = []
    labels = []
    # sh = wb.sheet_names()
    sh = wb.sheet_by_name("1_1 - 1_2 Survivor Corp")
    # iterate through excel and display data
    for i in range(sh.nrows):

      #print(sh.cell_value(i,0))
      patternRecogniton(sh.cell_value(i,0))
      #fname = "C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/Symptoms_1_1 - 1_2 Survivor Corp.csv"
      singularSymptoms.append(symptomBag)
      texts.append(sh.cell_value(i,0))
      #texts.append(" ".join(symptomBag))
      #labels.append(sh.cell_value(i,1))
      finalSymptoms += symptomBag
      symptomBag = []
    #exit()
    print("Start: ")
    import collections
    counter = collections.Counter(finalSymptoms)
    top50 = counter.most_common(50)
    # plotting high freq words
    import matplotlib.pyplot as plt
    high_Freq_words = pd.DataFrame(counter.most_common(50), columns=['words', 'count'])
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot horizontal bar graph
    high_Freq_words.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple")

    ax.set_title("Common Words Found in All Posts")
    plt.show()
    #exit()
    #word2vec.printInputData(texts, labels)
    #algo.printInputData(texts, labels)
    #neuralN.neuralNetworkClassification(texts, labels)
    #neuralN1.neuralNetworkClassification1(texts, labels)
    print("complete")
    #exit()
    print("Check")
    #print(finalSymptoms)
    #for words in finalSymptoms:
    #  print(",".join([i for i in words]))
    file1 = open("C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/FrequenceWords_1_1 - 1_2 Survivor Corp.txt","w")
    highFreq = []

    #for ele, cnt in counter.items():
    #  if cnt > 1 and ele not in s:
    #    highFreq.append((ele, cnt))
    #highFreq = sorted(highFreq, reverse=True)

    # highFreq = [counter.items() for ele, cnt in counter.items() if cnt > 1]

    # with open('C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/Symptoms_1_1 - 1_2 Survivor Corp.csv', 'w') as outfile:
    #    writer = csv.writer(outfile)
    cluster = []
    for row in top50:
        # writer.writerow(str(row[0]))
        #file1.wr
        # itelines(",".join([i for i in row]))
        file1.writelines(str(row[0])+ " ," + str(row[1]))
        cluster.append((str(row[0])))
        file1.writelines("\n")
    file1.close()
    print(cluster)
    print("Finished")


    '''Get co-appearance counts and scene counts'''
    appearance_count = {}
    symptom_count = {}
    appearance_count_0 = {}
    #appearance_count_1 = {}
    #appearance_count_2 = {}
    #appearance_count_3 = {}
    for i in top50:
      symptom_count[i[0]] = i[1]
      appearance_count_0[i[0]] = {}
      #appearance_count_1[i[0]] = {}
      #appearance_count_2[i[0]] = {}
      #appearance_count_3[i[0]] = {}
      for j in top50:
        if i != j:
          appearance_count_0[i[0]][j[0]] = 0
          #appearance_count_1[i[0]][j[0]] = 0
          #appearance_count_2[i[0]][j[0]] = 0
          #appearance_count_3[i[0]][j[0]] = 0
    '''Verifying the output format'''
    # print(appearance_count)
    # print(singularSymptoms)
    #print(symptom_count)
    cluster0 = []
    #cluster1 = []
    #cluster2 = []
    #cluster3 = []

    overlappingNodes = {}

    for symptomList,i in zip(singularSymptoms, label):
      cluster0.append(symptomList)
      #if i == 0:
      #    cluster0.append(symptomList)
      #if i == 1:
      #    cluster1.append(symptomList)
      #if i == 2:
      #    cluster2.append(symptomList)
      #if i == 3:
      #    cluster3.append(symptomList)
      for symptom in symptomList:
        for co_symptom in symptomList:
          if co_symptom != symptom:
              if co_symptom in appearance_count_0.keys():
                  if symptom in appearance_count_0[co_symptom].keys():
                      appearance_count_0[co_symptom][symptom] += 1

                #if symptom in appearance_count_3.keys():
                #  if co_symptom in appearance_count_3[symptom].keys():
                #    appearance_count_3[symptom][co_symptom] += 1
    #print(appearance_count)

    for node in cluster:
        cnt = 0
        for nodeList in cluster0:
            if node in nodeList:
                cnt += 1
                break

        #if cnt > 1:
        overlappingNodes[node] = cnt
    print(overlappingNodes)

    print("cluster0:",cluster0)
    print("node cluster0:", appearance_count_0)



    '''Building a Network Graph'''
    import plotly.offline as py
    import plotly.graph_objects as go
    import networkx as nx

    networkGraph = nx.Graph()


    '''Add node for each character'''
    for char in symptom_count.keys():
        if symptom_count[char] > 0:
            networkGraph.add_node(char, size = symptom_count[char])

    networkGraph.nodes
    # print(networkGraph.nodes)


    '''For each co-appearance between two characters, add an edge'''
    for symp in appearance_count_0.keys():
      for co_symp in appearance_count_0[symp].keys():
        if appearance_count_0[symp][co_symp] > 0:
          networkGraph.add_edge(symp, co_symp, weight=appearance_count_0[symp][co_symp], color='blue')

    networkGraph.edges()
    # print(networkGraph.edges())

    # Get positions for the nodes in G
    pos_ = nx.spring_layout(networkGraph)


    def make_edge(x, y, text, width):
      '''Creates a scatter trace for the edge between x's and y's with given width

      Parameters
      ----------
      x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])

      y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])

      width: the width of the line

      Returns
      -------
      An edge trace that goes between x0 and x1 with specified width.
      '''
      return go.Scatter(x=x,
                        y=y,
                        line=dict(width=width),
                        hoverinfo='text',
                        text=([text]),
                        mode='lines')


    '''For each edge, make an edge_trace, append to list'''
    edge_trace = []

    for edge in networkGraph.edges():
        #print("edge", edge)
        if networkGraph.edges()[edge]['weight'] > 0:
            #char_0 = (edge[1]).split()
            char_1 = edge[0]
            char_2 = edge[1]
            #color = 'cornflowerblue'
            x0, y0 = pos_[char_1]
            x1, y1 = pos_[char_2]

            text = char_1 + '--' + char_2 + ': ' + str(networkGraph.edges()[edge]['weight'])

            trace = make_edge([x0, x1, None], [y0, y1, None], text,
                              0.1 * networkGraph.edges()[edge]['weight'] ** 0.7)

            edge_trace.append(trace)

    # Make a node trace
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            textposition = "top center",
                            textfont_size = 10,
                            mode      = 'markers+text',
                            hoverinfo = 'none',
                            marker    = dict(color = [],
                                             size  = [],
                                             line  = None))
    '''For each node in midsummer, get the position and size and add to the node_trace'''
    for node in networkGraph.nodes():
        x, y = pos_[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple(['cornflowerblue'])
        node_trace['marker']['size'] += tuple([0.75*networkGraph.nodes()[node]['size']])
        node_trace['text'] += tuple(['<b>' + node + '</b>'])


    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    fig = go.Figure(layout = layout)

    for trace in edge_trace:
        fig.add_trace(trace)

    fig.add_trace(node_trace)

    fig.update_layout(showlegend = False)

    fig.update_xaxes(showticklabels = False)

    fig.update_yaxes(showticklabels = False)

    fig.show()
    py.plot(fig, filename='symptomNetwork1.html')
def kmeans_cluster_terms(num_clusters, top_n):
    """Performs K-means clustering and returns top_n features in each cluster.

    Args:
        num_cluster: k in k-means.
        top_n: top n features closest to the centroid of each cluster.

    Returns:
        cluster_centers: centroids of each cluster.
        distortion: sum of squares within each cluster.
        key_terms: list of top_n features closest to each centroid.
        labels: cluster assignments
    """
    # Generate cluster centers through the kmeans function
    cluster_centers, distortion = kmeans(feature_matrix.todense(), num_clusters)

    # Generate terms from the tfidf_vectorizer object
    terms = vectorizer.get_feature_names()

    # Display the top_n terms in that cluster
    key_terms = []
    for i in range(num_clusters):
        # Sort the terms and print top_n terms
        center_terms = dict(zip(terms, list(cluster_centers[i])))
        sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
        key_terms.append(sorted_terms[:top_n])

    # label the clusters
    labels, _ = vq(feature_matrix.todense(), cluster_centers, check_finite=True)

    return cluster_centers, distortion, key_terms, labels


'''ELbow method'''

# vary k from 2,10
distortions = []
centroids = []
top_10 = []
cluster_labels = []

num_clusters = range(2, 10)

for i in num_clusters:
    cluster_centers, distortion, key_terms, labels = kmeans_cluster_terms(i, 10)

    centroids.append(cluster_centers)
    distortions.append(distortion)
    top_10.append(key_terms)
    cluster_labels.append(labels)

# plot the elbow plot
elbow_plot_data = pd.DataFrame({'num_clusters': num_clusters,
                               'distortions': distortions})

sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot_data)
plt.title("Elbow Method")
#plt.show()

import predefinedFunctionsCluster as predef

'''Visiualize K means clustering'''

from sklearn.decomposition import PCA
pca = PCA()
components = pca.fit_transform(feature_matrix.todense())

xs, ys = components[:, 0], components[:, 1]
k = 2
print(k)
labels_two = list(cluster_labels[0])
#print(labels_two)
passlabels(labels_two)

exit()
'''k = 4'''
if k == 4:
    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}  #, 4: '#66a61e'}

    #set up cluster names using a dict
    cluster_names = {0: ', '.join(top_10[2][0]),
                     1: ', '.join(top_10[2][1]),
                     2: ', '.join(top_10[2][2]),
                     3: ', '.join(top_10[2][3])}

    # get the cluster labels
    labels_four = list(cluster_labels[2])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1 = predef.plot_kmeans(4, xs, ys, labels_four, cluster_names, cluster_colors, fig, ax1, len(xs))
    ax2 = predef.plot_silhouette(4, feature_matrix.todense(), labels_four, cluster_colors, fig, ax2)

    plt.suptitle("Clustering using K-means with K=4", fontsize=14, fontweight='bold')
    #plt.show()



if k == 2:
    # set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02'}  # , 2: '#7570b3', 3: '#e7298a'}  #, 4: '#66a61e'}

    # set up cluster names using a dict
    cluster_names = {0: ', '.join(top_10[0][0]),
                     1: ', '.join(top_10[0][1])}

    # get the cluster labels
    labels_two = list(cluster_labels[0])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1 = predef.plot_kmeans(2, xs, ys, labels_two, cluster_names, cluster_colors, fig, ax1, len(xs))
    ax2 = predef.plot_silhouette(2, feature_matrix.todense(), labels_two, cluster_colors, fig, ax2)

    plt.suptitle("Clustering using K-means with K=2", fontsize=14, fontweight='bold')
    #plt.show()

import patternRe
print("init")
patternRe.passlabels(labels_two)

if k == 3:
    # set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}  # , 3: '#e7298a'}  #, 4: '#66a61e'}

    # set up cluster names using a dict
    cluster_names = {0: ', '.join(top_10[1][0]),
                     1: ', '.join(top_10[1][1]),
                     2: ', '.join(top_10[1][2])}

    # get the cluster labels
    labels_three = list(cluster_labels[1])

    # set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}  # , 3: '#e7298a'}  #, 4: '#66a61e'}

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1 = predef.plot_kmeans(3, xs, ys, labels_three, cluster_names, cluster_colors, fig, ax1, len(xs))
    ax2 = predef.plot_silhouette(3, feature_matrix.todense(), labels_three, cluster_colors, fig, ax2)

    plt.suptitle("Clustering using K-means with K=3", fontsize=14, fontweight='bold')
    plt.show()


'''hierarchical clustering'''

method = "ward"

# Import cosine_similarity to calculate similarity
# Import modules necessary to plot dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
# Calculate the similarity distance
similarity_distance = 1 - cosine_similarity(feature_matrix)
if method == "complete":
    # Create mergings matrix
    mergings = linkage(similarity_distance, method='complete')

    # Plot the dendrogram, using title as label column
    dendrogram_ = dendrogram(mergings, orientation='top',
                   labels=[x for x in range(len(allposts))],
                   leaf_font_size=16
    )

    # Adjust the plot
    fig = plt.gcf()
    _ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
    fig.set_size_inches(108, 50)

    # Show the plotted dendrogram
    plt.show()
from scipy.cluster.hierarchy import ward
if method == "ward":
    linkage_matrix = ward(similarity_distance)

    # Plot the dendrogram, using title as label column
    dendrogram_ = dendrogram(linkage_matrix,
                             labels=[x for x in range(len(allposts))],
                             leaf_rotation=90,
                             leaf_font_size=16,
                             )

    # Adjust the plot
    fig = plt.gcf()
    _ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
    fig.set_size_inches(108, 21)

    plt.show()
'''with ward linkage method'''
from scipy.cluster.hierarchy import fcluster
h_ward_cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)

#fig, ax = plt.subplots(figsize=(17, 9))

ax2 = sns.scatterplot(x = xs, y=ys, hue=h_ward_cluster_labels, palette="Set2",
                      alpha=0.8, legend="full")
ax2.set_title("Hierarchical clustering using ward linkage with 4 clusters")
ax2.set_xlabel("First principal component")
ax2.set_ylabel("Second principal component")

ax1 = predef.plot_silhouette_hierarchical(2, feature_matrix.todense(), h_ward_cluster_labels, fig, ax1)

plt.suptitle("Clustering user movie reviews using AGNES ward linkage with 4 clusters",
             fontsize=14, fontweight='bold')
plt.show()


