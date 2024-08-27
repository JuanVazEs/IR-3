import re
from nltk.tokenize import word_tokenize

usr_pattern = re.compile(r'@\w+')
url_pattern = re.compile(r'http\S+|www\S+|https\S+')
num_pattern = re.compile(r'\b\d+\b')
diac_pattern = re.compile(r'[^\x00-\x7F]+')

def preprocess_text(text, group_usr=True, group_url=True, group_num=True, del_diac=True, lc=True):
    if del_diac:
        text = diac_pattern.sub('', text)  # Eliminar diacríticos
    if lc:
        text = text.lower()  # Convertir a minúsculas
    if group_usr:
        text = usr_pattern.sub('user', text)  # Reemplazar nombres de usuario por 'user'
    if group_url:
        text = url_pattern.sub('url', text)  # Reemplazar URLs por 'url'
    if group_num:
        text = num_pattern.sub('number', text)  # Reemplazar números por 'number'
    return text

def tokenize_and_preprocess(corpus, textconfig):
    processed_corpus = [preprocess_text(doc, **textconfig) for doc in corpus]
    return [word_tokenize(doc) for doc in processed_corpus]
