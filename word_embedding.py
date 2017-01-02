# -*- coding: utf-8 -*-
# @Author: ThomasO
import json
from gensim.models import Word2Vec
import spacy


class SentenceGenerator(object):
    """ """

    def __init__(self, annotations_path):
        """ """
        self.data = json.loads(annotations_path)
        self.nlp = spacy.load("en")

    def __iter__(self):
        """ """
        for item in self.data["annotations"]:
            p = self.nlp(item["caption"])
            yield [token.lemma_ for token in p if not token.is_punct]


if __name__ == '__main__':
    """ """
    path = "/Users/thomasopsomer/data/mscoco/annotations/captions_train2014.json"
    output = ""
    sentences = SentenceGenerator(path)
    # train model
    model = Word2Vec(sentences)
    # save model
    model.save(output)

