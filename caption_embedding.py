# -*- coding: utf-8 -*-
# @Author: ThomasO

import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, DocvecsArray
from gensim.models.phrases import Phrases
import spacy
import logging
import numpy as np
from gensim import matutils
import os
from collections import OrderedDict, defaultdict


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


def pre_clean_document(data, mwe=False, stopword=False):
    """ """
    print("Cleaning data (lemmatization) ...")
    nlp = spacy.load("en", parser=False, entity=False)
    clean_data = []
    for k in xrange(len(data)):
        raw = data.pop()
        p = nlp(raw["caption"], parse=False, entity=False)
        if stopword:
            c = [token.lemma_ for token in p if not token.is_stopword]
        else:
            c = [token.lemma_ for token in p]
        clean_data.append((raw["image_id"], c))
    del nlp
    print("Data cleand.")
    if mwe:
        print("Learning multiwords phrases...")
        bigram = Phrases(clean_data)
        for k in xrange(len(clean_data)):
            clean_data[k][1] = bigram[clean_data[k][1]]
        print("Clean data updated with learned bigram")
    return clean_data


def most_similar_to_word(model, word, topn=10, clip_start=0, clip_end=None):
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


def build_caption_emb_matrix_PV(model, iid_to_index, n_image):
    """
    """
    size = model.vector_size
    caption_emb_matrix = np.zeros(shape=(n_image, size), dtype=float)
    for image_id, index in iid_to_index.iteritems():
        image_caption_emb = model.docvecs[image_id]
        caption_emb_matrix[index] = image_caption_emb
    return caption_emb_matrix


def build_caption_emb_matix_W2V(model, doc_gen, iid_to_index, n_image):
    """
    Compute doc vector for each image using
    the mean of it caption words
    """
    # aggregate words for each image
    # because image have several captions
    d = defaultdict(list)
    for doc in doc_gen:
        image_id = doc.tags[0]
        d[image_id].extend(doc.words)
    # build vectors for each image using all its words
    size = model.vector_size
    caption_emb_matrix = np.zeros(shape=(n_image, size), dtype=float)
    #
    for image_id, index in iid_to_index.iteritems():
        image_caption_emb = np.mean([model[w] for w in d[image_id] if w in model], axis=0)
        caption_emb_matrix[index] = image_caption_emb
    return caption_emb_matrix


def build_image_id_to_index(mscoco_data_path, n_image):
    """ """
    iid_to_index = {}
    k = 0
    for fname in os.listdir(mscoco_data_path):
        if k >= n_image:
            break
        if fname.endswith(".jpg"):
            image_id = int(fname[:-4].split("_")[-1])
            iid_to_index[image_id] = k
            k += 1
    return iid_to_index


# avg_docvecs = DocvecsArray()
# avg_docvecs.reset_weights(model)
# for image_id, words in d.iteritems():
#     avg_docvecs.doctag_syn0[image_id] = np.mean([model[w] for w in words],
#                                                 axis=0)

if __name__ == '__main__':

    path = "/Users/thomasopsomer/data/mscoco/annotations/captions_train2014.json"
    mscoco_data_path = "/Users/thomasopsomer/data/mscoco/train2014"
    # init doc gen
    documents = DocumentGenerator(path)

    # learn doc vector as well as word vectors
    model = Doc2Vec(documents, size=128, min_count=2, iter=10, dbow_words=1,
                    dm=0, worker=4)
    # document most similar to a word
    most_similar_to_word(model, "dog")

    # extract matrix of vector in same order as image
    n_image = 80000
    size = 128
    doc2vec_path = "/Users/thomasopsomer/data/mscoco/caption_embedding/doc2vec.model"
    model = Doc2Vec.load(doc2vec_path)

    iid_to_index = build_image_id_to_index(mscoco_data_path, n_image)

    caption_emb_mat = build_caption_emb_matrix_PV(model, mscoco_data_path, n_image)
    caption_emb_mat_avg_w2v = build_caption_emb_matix_W2V(model, documents, iid_to_index, n_image)





