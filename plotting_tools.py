import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from collections import Counter

# Graficar la ley de Zipf
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

# Ley de Heaps
def heaps_law(corpus, model):
    unique_tokens = set()
    X, Y = [0], [0]
    for text in corpus:
        tokens = model['dictionary'].doc2bow(word_tokenize(text))
        unique_tokens.update([token_id for token_id, _ in tokens])
        X.append(X[-1] + len(text))
        Y.append(len(unique_tokens))
    return X, Y

def plot_heaps(corpus, V1, V2, label1, label2):
    X, Y = heaps_law(corpus, V1)
    plt.plot(X, Y, label=label1)
    X, Y = heaps_law(corpus, V2)
    plt.plot(X, Y, label=label2)
    plt.title('Ley de Heaps')
    plt.legend()
    plt.show()
