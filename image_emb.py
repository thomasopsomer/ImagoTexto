# -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
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


@begin.start
@begin.convert(nb_image=int, batch_size=int)
def extract_feature_vgg16(image_folder, nb_image, output_path=None,
                          batch_size=32, layer_name="fc1"):
    """ """
    # load VGG16 and weights
    base_model = VGG16(weights='imagenet', include_top=True)
    # Show the layer
    model = Model(input=base_model.input,
                  output=base_model.get_layer(layer_name).output)

    # instantiate Image generator with normalization of images
    datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True
    )

    train_generator = datagen.flow_from_directory(
        image_folder,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None)

    features = model.predict_generator(train_generator, 10)

    if output_path:
        # save the output as a Numpy array
        np.save(open(output_path, 'w'), features)
        return
    else:
        return features


# img_path = "/Users/thomasopsomer/github/ImagoTexto/overfeat/test.jpg"
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# import time
# t = time.time()
# features = model.predict(x)
# print(time.time() - t)
# print features.shape


