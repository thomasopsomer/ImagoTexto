# -*- coding: utf-8 -*-
# @Author: ThomasO

import numpy as np
from cca import cca
from gensim.models.doc2vec import Doc2Vec
from gensim import matutils
from math import sqrt
from sklearn.preprocessing import normalize


class SearchJointEmbedding(object):
    """ """
    def __init__(self, caption_emb_path, image_emb_path, iid_to_index,
                 doc2vec_path, p=4, W_caption=None, W_image=None, D=None):
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
        self.doc2vec = Doc2Vec.load(doc2vec_path)

        if all([x is None for x in [W_caption, W_image, D]]):
            self.d = 128
            self.W_caption, self.W_image, self.D = \
                cca(self.caption_emb, self.image_emb, d=128)
        elif all([x is not None for x in [W_caption, W_image, D]]):
            self.W_caption = W_caption
            self.W_image = W_image
            self.D = D
            self.d = len(D)
        # normalize cca with eigvenvalue at power p
        self.W_caption = np.dot(self.W_caption, np.diag(self.D) ** p)
        self.W_image = np.dot(self.W_image, np.diag(self.D) ** p)

    def t2i(self, words, pv=False, topn=10):
        """ """
        # get words vectors
        if isinstance(words, basestring):
            v = self.doc2vec[words]
        else:
            if pv:
                v = self.doc2vec.infer_vector(pv)
            else:
                v = np.mean([self.doc2vec[w] for w in words], axis=0)
        # get joint embedding for the word
        jv = np.dot(v, self.W_caption[:])
        jv = matutils.unitvec(jv)

        # nearest neighbor search
        dists = np.dot(self.joint_image_emb, jv.flatten())
        best = matutils.argsort(dists, topn=topn, reverse=True)
        result = [(self.index_to_iid[sim], float(dists[sim])) for sim in best]
        return result

    def t2t(self, words, pv=False, topn=10):
        """ """
        # get words vectors
        if isinstance(words, basestring):
            v = self.doc2vec[words]
        else:
            if pv:
                v = self.doc2vec.infer_vector(pv)
            else:
                v = np.mean([self.doc2vec[w] for w in words], axis=0)
        # get joint embedding for the word
        jv = np.dot(v, self.W_caption[:])
        jv = matutils.unitvec(jv)

        # nearest neighbor search
        dists = np.dot(self.joint_caption_emb, jv.flatten())
        best = matutils.argsort(dists, topn=topn, reverse=True)
        result = [(self.index_to_iid[sim], float(dists[sim])) for sim in best]
        return result

    def i2t(self, image_id):
        pass

    def index(self, remove_original_emb, p=4):
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
            # precompute l2 norm
            self.joint_caption_emb[i, :] /= sqrt((self.joint_caption_emb[i, :] ** 2).sum(-1))
            self.joint_image_emb[i, :] /= sqrt((self.joint_image_emb[i, :] ** 2).sum(-1))

        if remove_original_emb:
            del self.caption_emb
            del self.image_emb
        return


def joint(a, W):
    pass



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



