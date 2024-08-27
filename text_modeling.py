# text_modeling.py

import gc
from gensim import corpora, models
from gensim.models import Phrases
import matplotlib.pyplot as plt
from collections import Counter

# Importar tokenize_and_preprocess desde text_processing.py
from text_processing import tokenize_and_preprocess

def create_qgrams(text, q=3):
    qgrams = []
    for word in text:
        word = f'^{word}$'  # Agregar marcadores de inicio y fin a la palabra
        qgrams.extend([word[i:i+q] for i in range(len(word) - q + 1)])
    return qgrams

def text_model_and_vectors(corpus, textconfig=None, ngram_range=(1, 1), q=3):

    if textconfig is None:
        textconfig = {
            'group_usr': True,
            'group_url': True,
            'group_num': True,
            'del_diac': True,
            'lc': True
        }

    # Preprocesar y tokenizar
    tokenized_corpus = tokenize_and_preprocess(corpus, textconfig)

    # Aplicar n-gramas (unigrama, bigrama, trigrama)
    if ngram_range[1] > 1:
        bigram_phraser = Phrases(tokenized_corpus, min_count=10, threshold=15)
        trigram_phraser = Phrases(bigram_phraser[tokenized_corpus], threshold=15)
        tokenized_corpus = [trigram_phraser[bigram_phraser[doc]] for doc in tokenized_corpus]

    # Aplicar q-gramas
    qgram_corpus = [create_qgrams(doc, q=5) for doc in tokenized_corpus]

    # Combinar el corpus tokenizado con q-gramas
    combined_corpus = [doc + qgram_doc for doc, qgram_doc in zip(tokenized_corpus, qgram_corpus)]

    # Crear diccionario y corpus (representación dispersa)
    dictionary = corpora.Dictionary(combined_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in combined_corpus]

    # Modelo TF-IDF con suavizado
    tfidf_model = models.TfidfModel(doc_term_matrix, dictionary=dictionary, smartirs="ltc")
    tfidf_vectors = [tfidf_model[doc] for doc in doc_term_matrix]

    # Liberar memoria
    del tokenized_corpus
    del qgram_corpus
    gc.collect()

    return {
        'textconfig': textconfig,
        'vectors': tfidf_vectors,
        'dictionary': dictionary,
        'tfidf_model': tfidf_model
    }

def vectorize_corpus(model, corpus):
    return [model['tfidf_model'][model['dictionary'].doc2bow(word_tokenize(doc))] for doc in corpus]

def plot_zipf(V1, V2, label1, label2):
    count1 = Counter([token_id for doc in V1['vectors'] for token_id, _ in doc])
    count2 = Counter([token_id for doc in V2['vectors'] for token_id, _ in doc])
    sorted_counts1 = sorted(count1.values(), reverse=True)
    sorted_counts2 = sorted(count2.values(), reverse=True)
    plt.plot(sorted_counts1, label=label1)
    plt.plot(sorted_counts2, label=label2)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Ley de Zipf')
    plt.show()
