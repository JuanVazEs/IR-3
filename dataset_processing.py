import gzip
import json
import os
import requests

# Procesar conjuntos de datos
def process_datasets():
    def create_dataset(corpusfile, textkey, labelkey):
        text, labels = [], []
        with gzip.open(corpusfile, 'rt') as f:
            for line in f:
                r = json.loads(line)
                labels.append(r[labelkey])
                text.append(r[textkey])
        return {"text": text, "labels": labels}

    def get_dataset(dbname):
        os.makedirs("data", exist_ok=True)
        dbfile = os.path.join("data", dbname)
        baseurl = "https://github.com/sadit/TextClassificationTutorial/raw/main/data"
        if not os.path.isfile(dbfile):
            url = f"{baseurl}/{dbname}"
            r = requests.get(url)
            with open(dbfile, 'wb') as f:
                f.write(r.content)
        return dbfile

    def read_news():
        train = "spanish-twitter-news-and-opinions-top25-68.train.json.gz"
        test = "spanish-twitter-news-and-opinions-top25-68.test.json.gz"
        return (create_dataset(get_dataset(train), "text", "screen_name"),
                create_dataset(get_dataset(test), "text", "screen_name"))

    # Llamada a la función principal para obtener los datos
    D, Q = read_news()
    return D, Q
