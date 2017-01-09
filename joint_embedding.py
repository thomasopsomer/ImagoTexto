# -*- coding: utf-8 -*-
# @Author: ThomasO

import numpy as np
from cca import cca
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim import matutils
from math import sqrt
from sklearn.preprocessing import normalize

# image embedding dependency
try:
    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing.image import (
        array_to_img, img_to_array, load_img
    )
except:
    print("Keras not found, no cnn features on the fly.")
# nlp preprrocessing
import spacy


class SearchJointEmbedding(object):
    """ """
    def __init__(self, caption_emb_path, image_emb_path, iid_to_index,
                 d2v_path=None, w2v_path=None,
                 p=None, W_caption=None, W_image=None, D=None,
                 cnn=None, nlp=False):
        """ """
        self.caption_emb = normalize(np.load(caption_emb_path))
        # self.caption_emb = normalize(self.caption_emb)
        self.image_emb = normalize(np.load(image_emb_path))
        # self.image_emb = normalize(self.image_emb)
        # check dimension agrees
        assert self.caption_emb.shape[0] == self.image_emb.shape[0]
        self.n = self.caption_emb.shape[0]
        #
        self.caption_dim = self.caption_emb.shape[1]
        self.image_dim = self.image_emb.shape[1]

        self.iid_to_index = iid_to_index
        self.index_to_iid = sorted(self.iid_to_index.keys())
        # load word2vec or doc2vec model for word vectors
        if w2v_path is not None:
            self.doc2vec = Word2Vec.load(w2v_path)
        elif d2v_path is not None:
            self.doc2vec = Doc2Vec.load(d2v_path)
        else:
            raise ValueError("Need at least a doc2vec or word2ve model path")

        if all([x is None for x in [W_caption, W_image, D]]):
            self.d = 128
            self.W_caption, self.W_image, self.D = \
                cca(self.caption_emb, self.image_emb, d=128)
        elif all([x is not None for x in [W_caption, W_image, D]]):
            self.W_caption = W_caption
            self.W_image = W_image
            self.D = D
            self.d = len(D)

        if p is not None:
            # normalize cca with eigvenvalue at power p
            self.W_caption = np.dot(self.W_caption, np.diag(self.D) ** p)
            self.W_image = np.dot(self.W_image, np.diag(self.D) ** p)

        # load cnn if needed to perform i2t on new images on the fly
        if cnn == "vgg16":
            self.cnn = VGG16(weights='imagenet', include_top=True)
            self.target_size = (224, 224)
        elif cnn == "vgg19":
            self.cnn = VGG19(weights='imagenet', include_top=True)
            self.target_size = (224, 224)

        # load spacy if want to preproessing query
        if nlp:
            self.nlp = spacy.load("en", parser=False, entity=False)

    def t2i(self, words=None, pv=False, topn=10, positive=None, negative=None):
        """ """
        # get words vectors
        jv = self.get_words_vector(words, positive=None, negative=None, pv=pv)

        # nearest neighbor search
        dists = np.dot(self.joint_image_emb, jv.flatten())
        best = matutils.argsort(dists, topn=topn, reverse=True)
        result = [(self.index_to_iid[sim], float(dists[sim])) for sim in best]
        return result

    def t2t(self, words, pv=False, topn=10):
        """ """
        # get words vectors
        jv = self.get_words_vector(words, pv=pv)
        # nearest neighbor search
        dists = np.dot(self.joint_caption_emb, jv.flatten())
        best = matutils.argsort(dists, topn=topn, reverse=True)
        result = [(self.doc2vec.index2word[sim], float(dists[sim])) for sim in best]
        return result

    def i2t(self, image_id, topn=10, vector):
        """ """
        # get image vector
        jv = self.get_image_vector(img_id=image_id)
        # nearest neighbor search
        dists = np.dot(self.joint_word_emb, jv.flatten())
        best = matutils.argsort(dists, topn=topn, reverse=True)
        result = [(self.doc2vec.index2word[sim], float(dists[sim])) for sim in best]
        return result

    def get_words_vector(self, words=None, pv=False, positive=[], negative=[]):
        """ """
        # get words vectors
        if words is not None:
            if isinstance(words, basestring):
                if hasattr(self, "nlp"):
                    p = self.nlp(words, parse=False, entity=False)
                    words = [tok.lemma_ for tok in p]
                else:
                    words = words.split(" ")
            if len(words) == 1:
                v = self.doc2vec[words[0]]
            else:
                if pv:
                    v = self.doc2vec.infer_vector(words)
                else:
                    v = np.mean([self.doc2vec[w] for w in words], axis=0)
        elif positive or negative:
            s = []
            s.extend([self.doc2vec[w] for w in positive])
            s.extend([-self.doc2vec[w] for w in negative])
            v = np.sum(s)
        # get joint embedding for the word
        jv = np.dot(v, self.W_caption[:])
        jv = matutils.unitvec(jv)
        return jv

    def get_image_vector(self, img_id=None, img_path=None):
        """ """
        if img_path is not None:
            if hasattr(self, "cnn"):
                img = image.load_img(img_path, target_size=self.target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                img_emb = np.dot(self.cnn.predict(x), self.W_image)
                return img_emb
            else:
                raise ValueError("Need to load cnn")
        elif img_id is not None:
            if img_id in self.iid_to_index:
                img_emb = self.joint_image_emb[self.iid_to_index[img_id]]
                return img_emb
            else:
                raise KeyError("img_id %s not im train dataset" % img_id)
        else:
            raise ValueError("Need at list an img_id or an img_path")

    def index(self, remove_original_emb, p=4, norm=True):
        """
        precompute representation of image and caption in the joint embedding
        """
        self.joint_caption_emb = np.zeros(shape=(self.n, self.d), dtype=float)
        self.joint_image_emb = np.zeros(shape=(self.n, self.d), dtype=float)
        for i in xrange(self.n):
            # compute joint embedding
            self.joint_caption_emb[i] = np.dot(self.caption_emb[i],
                                               self.W_caption).T
            self.joint_image_emb[i] = np.dot(self.image_emb[i],
                                             self.W_image).T
            if norm:
                # precompute l2 norm
                self.joint_caption_emb[i, :] /= sqrt((self.joint_caption_emb[i, :] ** 2).sum(-1))
                self.joint_image_emb[i, :] /= sqrt((self.joint_image_emb[i, :] ** 2).sum(-1))

        # precompute joint representation of words
        self.joint_word_emb = np.zeros(shape=self.doc2vec.wv.syn0.shape, dtype=float)
        for k in xrange(self.doc2vec.syn0.shape[0]):
            self.joint_word_emb[k] = np.dot(self.doc2vec.wv.syn0[k],
                                            self.W_caption)
            if norm:
                # precompute l2 norm
                self.joint_word_emb[k, :] /= sqrt((self.joint_word_emb[k, :] ** 2).sum(-1))

        if remove_original_emb:
            del self.caption_emb
            del self.image_emb
        return


if __name__ == '__main__':

    from caption_embedding import build_image_id_to_index

    # input image_embedding, caption embedding

    caption_emb_path = "/Users/thomasopsomer/data/mscoco/caption_embedding/caption_emb_pv.npy"
    image_emb_path = "/Users/thomasopsomer/data/mscoco/image_embedding/mscoco_train_80k_vgg16_fc1.npy"
    image_emb_path = "/Users/thomasopsomer/data/mscoco/image_embedding/mscoco_vgg16_l_prediction.npy"
    doc2vec_path = "/Users/thomasopsomer/data/mscoco/caption_embedding/doc2vec.model"

    mscoco_data_path = "/Users/thomasopsomer/data/mscoco/train2014"
    n_image = 80000
    iid_to_index = build_image_id_to_index(mscoco_data_path, n_image)
    # image_emb = np.load(image_emb_path)
    # # check dimension agrees
    # assert caption_emb.shape[0] == image_emb.shape[0]

    # # compute projection matrix for each embedding toward the joint embedding
    # # using CCA
    # W_caption, W_image, D = cca(caption_emb, image_emb, d=128)

    search = SearchJointEmbedding(
        caption_emb_path=caption_emb_path,
        image_emb_path=image_emb_path,
        iid_to_index=iid_to_index,
        doc2vec_path=doc2vec_path)



