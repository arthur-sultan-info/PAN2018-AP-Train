# coding=utf-8
import re
import string
from nltk.corpus import stopwords



class clean_ar_txt:
    '''
        Tokenizer for arabic texts
    '''
    stopwords_list = stopwords.words('arabic')
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
    def __init__(self):
        pass

    def normalize_arabic(self, text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        return text


    def remove_diacritics(self, text):
        text = re.sub(clean_ar_txt.arabic_diacritics, '', text)
        return text


    def remove_punctuations(self, text):
        translator = str.maketrans('', '', clean_ar_txt.punctuations_list)
        return text.translate(translator)


    def remove_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    def tokenize(self, text):
        text = self.remove_punctuations(text)
        text = self.normalize_arabic(text)
        text = self.remove_diacritics(text)
        text = self.remove_repeating_char(text)
        return [x for x in text.split() if x not in clean_ar_txt.stopwords_list]
        #return text.split()
