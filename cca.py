# -*- coding: utf-8 -*-
# @Author: ThomasO
"""

Canonical correlation analysis can be defined as the problem of finding two sets of
basis vectors, one for x and the other for y, such that the correlations between the
projections of the variables onto these basis vectors are mutually maximized.


"""
import numpy as np
import scipy.linalg


def cca(word_emb, image_emb, d=128):
    """ """
    # get the embedding size of each representation
    word_emb_size = word_emb.shape[1]
    image_emb_size = image_emb.shape[1]
    # number of total dimension
    nc = word_emb_size + image_emb_size

    # min embedding size
    min_emb_size = min(word_emb_size, image_emb_size)

    # compute covariant block
    S_w_w = np.matmul(word_emb.T, word_emb)
    S_i_i = np.matmul(image_emb.T, image_emb)
    S_w_i = np.matmul(word_emb.T, image_emb)
    # assemble whole covariance matrix A
    A = np.bmat([[S_w_w, S_w_i], [S_w_i.T, S_i_i]])
    # assemble second member of generalize eigen problem
    B = np.bmat([[S_w_w, np.zeros(shape=(word_emb_size, image_emb_size))],
                [np.zeros(shape=(image_emb_size, word_emb_size)), S_i_i]])
    # solve generalize eigen problem
    eigvenvalues, eigenvectors = scipy.linalg.eigh(
        A, B, eigvals=(nc - min_emb_size, nc - 1))

    # keep d eigenvectors and attribute first word_emb_size composant
    # to the projection matrix for word embedding
    W_word = eigenvectors[0:word_emb_size, 0:d]
    # keep d eigenvectors and attribute the remaining composant
    # to the projection matrix for image embedding
    W_image = eigenvectors[word_emb_size:, 0:d]
    # keep the top d eigenvalue for later :)
    D = eigvenvalues[0:d]

    return W_word, W_image, D


if __name__ == '__main__':
    # test
    word_emb = np.random.random_sample(size=(80000, 128))
    image_emb = np.random.random_sample(size=(80000, 512))
    # perform cca
    W_w, W_i, D = cca(word_emb, image_emb)


