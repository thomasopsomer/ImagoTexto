# -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras_model.vgg16 import VGG16
from keras_model.imagenet_utils import preprocess_input
import numpy as np
from keras.models import Model
from keras.preprocessing.image import (
    ImageDataGenerator, array_to_img, img_to_array, load_img
)


# Predict feature using VGG16
base_model = VGG16(weights='imagenet', include_top=True)
model = Model(input=base_model.input,
              output=base_model.get_layer('fc1').output)

img_path = "/Users/thomasopsomer/github/ImagoTexto/overfeat/test.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# Predict feature with image generator
datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
)

train_generator = datagen.flow_from_directory(
        '/Users/thomasopsomer/data/mscoco/test',
        target_size=(224, 224),
        batch_size=2,
        class_mode=None)


features = model.predict_generator(train_generator, 10)


