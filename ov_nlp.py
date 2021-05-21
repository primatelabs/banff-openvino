#!/usr/bin/env python3

# Copyright 2004-2021 Primate Labs Inc.  All rights reserved.

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np
import matplotlib.pyplot as plt

from openvino.inference_engine import IECore, IENetwork

import PIL
from PIL import Image

import os
import sys

import bert
from bert.model import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from transformer import Transformer


def text_classification():
  h5_model = tf.keras.models.load_model('./text_classification/bert_tiny_f32.h5', 
              custom_objects={ 'BertModelLayer': BertModelLayer, 'tf':tf })
  os.system("mkdir temp")
  tf.saved_model.save(h5_model, "./temp")

  os.system("python ./model_optimizer/mo_tf.py \
            --saved_model_dir ./temp \
            --output_dir ./text_classification")
  os.system("rm -rf ./temp")

  src = np.zeros((1, 128), dtype=np.int32)

  ie = IECore()
  network = ie.read_network(model = "text_classification/saved_model.xml", weights = "text_classification/saved_model.bin")
  exec_net = ie.load_network(network=network, device_name='CPU')

  input_blob = next(iter(network.input_info.keys()))
  output_blob = next(iter(network.outputs))
  result = exec_net.infer(inputs={input_blob: src})


def machine_translation():
  eng_vocab_size = 14214
  fre_vocab_size = 27730
  eng_length = 49
  fre_length = 56
  num_layers = 4

  h5_model = Transformer(eng_vocab_size, fre_vocab_size, eng_length, 
                          fre_length, batch_size = 1, num_layers = num_layers)
  h5_model.load_weights("/Users/jzhu/Documents/Banff/banff-models/NLP/Machine_Translation/h5/Transformer_complex_f32.h5")
  os.system("mkdir temp")
  tf.saved_model.save(h5_model, "./temp")

  os.system("python ./model_optimizer/mo_tf.py \
            --saved_model_dir ./temp \
            --output_dir ./machine_translation")
  os.system("rm -rf ./temp")

  src_1 = np.zeros((1, 49), dtype=np.int32)
  src_2 = np.zeros((1, 55), dtype=np.int32)
  ie = IECore()
  network = ie.read_network(model = "machine_translation/saved_model.xml", weights = "machine_translation/saved_model.bin")
  exec_net = ie.load_network(network=network, device_name='CPU')

  input_blob = next(iter(network.input_info.keys()))
  output_blob = next(iter(network.outputs))
  result = exec_net.infer(inputs={input_blob: [src_1, src_2]})


def main():
  text_classification()
  machine_translation()


if __name__ == '__main__':
  main()
