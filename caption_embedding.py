# -*- coding: utf-8 -*-
# @Author: ThomasO

import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases
import spacy
import logging
import numpy as np
from gensim import matutils


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class DocumentGenerator(object):
    """ """
    def __init__(self, annotations_path, mwe=False):
        """ """
        print("Load json data...")
        with open(annotations_path) as f:
            data = json.load(f)["annotations"]
        print("Json data loaded")
        # clean sentences
        self.data = pre_clean_document(data, mwe)
    #
    def __iter__(self):
        """ """
        for image_id, caption in self.data:
            yield TaggedDocument(caption, [image_id])


def pre_clean_document(data, mwe=False):
    """ """
    print("Cleaning data (lemmatization) ...")
    nlp = spacy.load("en", parser=False, entity=False)
    clean_data = []
    for k in xrange(len(data)):
        raw = data.pop()
        p = nlp(raw["caption"], parse=False, entity=False)
        clean_data.append((raw["image_id"], [token.lemma_ for token in p]))
    del nlp
    print("Data cleand.")
    if mwe:
        print("Learning multiwords phrases...")
        bigram = Phrases(clean_data)
        for k in xrange(len(clean_data)):
            clean_data[k][1] = bigram[clean_data[k][1]]
        print("Clean data updated with learned bigram")
    return clean_data


path = "/Users/thomasopsomer/data/mscoco/annotations/captions_train2014.json"

documents = DocumentGenerator(path)

model = Doc2Vec(documents, size=128, min_count=2, iter=10, dbow_words=1,
                dm=0, worker=4)


def most_similar(model, word, topn=10, clip_start=0, clip_end=None):
    """
    """
    model.docvecs.init_sims()
    vector = model[word]
    clip_end = clip_end or len(model.docvecs.doctag_syn0norm)
    #
    dists = np.dot(model.docvecs.doctag_syn0norm[clip_start:clip_end], vector)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn, reverse=True)
    # ignore (don't return) docs from the input
    result = [(model.docvecs.index_to_doctag(sim), float(dists[sim])) for sim in best]
    return result[:topn]



most_similar(model, "dog")


