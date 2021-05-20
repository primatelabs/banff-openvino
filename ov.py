#!/usr/bin/env python3

# Copyright 2004-2021 Primate Labs Inc.  All rights reserved.

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np
import matplotlib.pyplot as plt

from openvino.inference_engine import IECore, IENetwork

import PIL
from PIL import Image


# Normalize image values from 0~255 to -1~1 for float32 and float16
def normalize(img):
  img = np.expand_dims(img / 255, 0)
  img = np.moveaxis(img, -1, 1)
  return img


def image_classification():
  h5_model = tf.keras.models.load_model('image_classification/mobilenet_v1_f32.h5')

  # TODO: Convert H5 to OpenVINO

  src = Image.open("ic-001.jpeg")
  src = src.resize((224, 224))
  src = normalize(np.array(src))

  ie = IECore()
  network = ie.read_network(model = "image_classification/saved_model.xml", weights = "image_classification/saved_model.bin")
  exec_net = ie.load_network(network=network, device_name='CPU')

  input_blob = next(iter(network.input_info.keys()))
  output_blob = next(iter(network.outputs))
  result = exec_net.infer(inputs={input_blob: src})
  score = result['StatefulPartitionedCall/mobilenet_1.00_224/predictions/Softmax'][0]
  klass = score.tolist().index(max(score))
  print(klass)


def image_segmentation():
  h5_model = tf.keras.models.load_model('image_segmentation/deeplabv3_mobilenetv2_f32.h5')

  # TODO: Convert H5 to OpenVINO

  ie = IECore()
  network = ie.read_network(model = "image_segmentation/saved_model.xml", weights = "image_segmentation/saved_model.bin")
  exec_net = ie.load_network(network=network, device_name='CPU')

  input_blob = next(iter(network.input_info.keys()))
  output_blob = next(iter(network.outputs))
  result = exec_net.infer(inputs={input_blob: src})
  score = result['StatefulPartitionedCall/mobilenet_1.00_224/predictions/Softmax'][0]
  print(score)
  klass = score.tolist().index(max(score))
  print(klass)



def main():
  image_classification()
  image_segmentation()


if __name__ == '__main__':
  main()
