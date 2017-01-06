# -*- coding: utf-8 -*-
# @Author: ThomasO
import json
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
import spacy
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class SentenceGenerator(object):
    """ """

    def __init__(self, annotations_path, mwe=False):
        """ """
        print("Load json data...")
        with open(annotations_path) as f:
            data = json.load(f)["annotations"]
        print("Json data loaded")
        # clean sentences
        self.data = pre_clean(data, mwe)

    def __iter__(self):
        """ """
        for sent in self.data:
            yield sent


def pre_clean(data, mwe=False):
    """ """
    print("Cleaning data (lemmatization) ...")
    nlp = spacy.load("en", parser=False, entity=False)
    clean_data = []
    for k in xrange(len(data)):
        raw = data.pop()
        p = nlp(raw["caption"], parse=False, entity=False)
        clean_data.append([token.lemma_ for token in p])
    del nlp
    print("Data cleand.")
    if mwe:
        print("Learning multiwords phrases...")
        bigram = Phrases(clean_data)
        for k in xrange(len(clean_data)):
            clean_data[k] = bigram[clean_data[k]]
        print("Clean data updated with learned bigram")
    return clean_data


if __name__ == '__main__':
    """ """
    path = "/Users/thomasopsomer/data/mscoco/annotations/captions_train2014.json"
    output = "/Users/thomasopsomer/data/mscoco/annotations/caption_train_emb.model"
    sentences = SentenceGenerator(path)
    # train model
    model = Word2Vec(sentences, iter=20, min_count=2, sg=1, size=128)
    # save model
    model.save(output)

