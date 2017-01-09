from caption_embedding import build_image_id_to_index
from joint_embedding import SearchJointEmbedding
import numpy as np


# input image_embedding, caption embedding

caption_emb_path = "/Users/thomasopsomer/data/mscoco/caption_embedding/caption_emb_pv.npy"
image_emb_path = "/Users/thomasopsomer/data/mscoco/image_embedding/mscoco_train_80k_vgg16_fc1.npy"
image_emb_path = "/Users/thomasopsomer/data/mscoco/image_embedding/mscoco_train_vgg16_pred.npy"
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
    d2v_path=doc2vec_path,
    p=2)


search.index(False)

search.t2i("airplane")



from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize

caption_emb = normalize(np.load(caption_emb_path))
image_emb = normalize(np.load(image_emb_path))


cca = CCA(n_components=100)
cca.fit(caption_emb, image_emb)

caption_emb_c, image_emb_c = cca.transform(caption_emb, caption_emb)


X = np.array([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])

S_w_w = np.matmul(X.T, Y)
S_i_i = np.matmul(Y.T, Y)
S_w_i = np.matmul(X.T, Y)
S_i_w = np.matmul(Y.T, X)


cca = CCA(n_components=2)
cca.fit(X, Y)

cca.x_weights_
cca.y_weights_


