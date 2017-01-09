# -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input
# from keras_model.imagenet_utils import preprocess_input
# from keras_model.vgg16 import VGG16
import numpy as np
from keras.models import Model
from keras.preprocessing.image import (
    ImageDataGenerator, array_to_img, img_to_array, load_img
)
try:
    import begin
except:
    import pip
    pip.main(["install", "begins"])
    import begin


def MyGenerator(folder_path, nb_image, batch_size, target_size):
    """ """
    import os
    flist = [os.path.join(folder_path, x) for x in os.listdir(folder_path)
             if x.endswith(".jpg")]
    print("Found %s images in folder %s" % (len(flist), folder_path))
    #
    while True:
        for k in xrange(0, nb_image, batch_size):
            # load files
            X = load_files(flist[k:k + batch_size], target_size)
            yield X


def load_files(file_list, target_size):
    """ """
    b = []
    for img_path in file_list:
        img = image.load_img(img_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        b.append(x)
    return np.asarray(b).reshape(len(file_list), target_size[0], target_size[1], 3)


@begin.start
@begin.convert(nb_image=int, batch_size=int)
def extract_feature_vgg16(image_folder, nb_image, output_path=None,
                          net="vgg16", batch_size=32, layer_name="predictions"):
    """
    `layer_name` in [fc1, fc2, prediction]
    `net`in [vgg16, vgg19, block5]
    """

    # load VGG16 and weights
    if net == "vgg16":
        base_model = VGG16(weights='imagenet', include_top=True)
        target_size = (224, 224)
    elif net == "vgg19":
        base_model = VGG19(weights='imagenet', include_top=True)
        target_size = (224, 224)

    # Show the layer
    if layer_name in ['fc1', 'fc2', 'predictions']:
        model = Model(input=base_model.input,
                      output=base_model.get_layer(layer_name).output)
    else:
        model = base_model
    # instantiate Image generator with normalization of images
    # datagen = ImageDataGenerator()

    # train_generator = datagen.flow_from_directory(
    #     image_folder,
    #     target_size=(224, 224),
    #     batch_size=batch_size,
    #     class_mode=None)
    data_generator = MyGenerator(image_folder, nb_image, batch_size,
                                 target_size)

    features = model.predict_generator(data_generator, nb_image)

    if layer_name == "block5":
        s = features.shape
        features = features.reshape(nb_image, s[1] * s[2] * s[3])

    if output_path:
        # save the output as a Numpy array
        np.save(open(output_path, 'wb'), features)
        return
    else:
        return features



# img_path = "/Users/thomasopsomer/github/ImagoTexto/overfeat/test.jpg"
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)


# folder_path = "/Users/thomasopsomer/data/mscoco/test/sub_test"
# g = MyGenerator(folder_path, nb_image=10, batch_size=10,
#                 target_size=(224, 224))

# features = model.predict_generator(g, 10)

# for k in range(features.shape[0]):
#     print('Predicted:', decode_predictions(features[k].reshape(1, 1000)))

# import time
# t = time.time()
# features = model.predict(x)
# print(time.time() - t)
# print features.shape


