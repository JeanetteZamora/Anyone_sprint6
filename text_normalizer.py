from operator import truediv
import re
import string
import spacy
import unicodedata
import string
import nltk
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')

    text = TAG_RE.sub('', text)

    return text


def stem_text(text):
    ps = PorterStemmer()

    words = tokenizer.tokenize(text)
    text = " ".join([ps.stem(w) for w in words])

    return text


def lemmatize_text(text):
    
    text = nlp(text)

    text =  " ".join([token.lemma_ for token in text])

    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    text = contractions_pattern.sub(expand_match, text)
    text = re.sub("'", "", text)

    return text


def remove_accented_chars(text):

    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    return text


def remove_special_chars(text, remove_digits=False):

    if remove_digits == True:
       text = re.sub('[0-9]+', '', text)

    else:
      text = re.sub('[^\da-zA-Z ]','', text)

    return text

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    
    token_word = tokenizer.tokenize(text.lower())

    text = [word for word in token_word if not word in stopwords]

    text = " ".join(text)

    return text


def remove_extra_new_lines(text):

    text = text.replace('\n', ' ')

    return text

def remove_extra_whitespace(text):

    text = " ".join(text.split())

    return text

def remove_puntuaction(text):

    text = ''.join([char for char in text if char not in string.punctuation])

    return text

    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=True,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    puntuaction = True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        #Remove puntuaction
        if puntuaction:
            doc = remove_puntuaction(doc)
            
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
