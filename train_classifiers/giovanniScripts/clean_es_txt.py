# coding=utf-8
import re
import string
from nltk.corpus import stopwords


class clean_es_txt:

    punctuations_list = string.punctuation+'¿'+'¡'
    stopwords_list = stopwords.words('spanish')

    def __init__(self):
        pass

    def normalize(self, text):
        return text

    def remove_punctuations(self, text):
        translator = str.maketrans('', '', clean_es_txt.punctuations_list)
        return text.translate(translator)

    def remove_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    def tokenize(self, text):
        text = self.remove_punctuations(text)
        #text = self.normalize(text)
        text = self.remove_repeating_char(text)
        return [x for x in text.split() if x not in clean_es_txt.stopwords_list]
        #return text.split()
