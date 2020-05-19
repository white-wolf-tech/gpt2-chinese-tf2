import tensorflow as tf
from model.gpt2 import TFGPT2Model
from model.gpt2_config import GPT2Config
import sys

checkpoint_path="./checkpoint/train"

config = GPT2Config()
gpt2_model = TFGPT2Model(config)
gpt2_model.load_weights(tf.train.latest_checkpoint(checkpoint_path))

converter = tf.lite.TFLiteConverter.from_keras_model(gpt2_model)
tflite_model = converter.convert()
open("chat.tflite", "wb").write(tflite_model)