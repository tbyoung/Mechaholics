import argparse
import sys
import time
import numpy as np
import cv2
import tensorflow as tf
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import numpy.random
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def run(model: str, num_threads: int, enable_edgetpu: bool) -> None:
  start_time = time.time()

  # Initialize the object classification model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  classification_options = processor.ClassificationOptions(
      max_results=2)
  options = vision.ImageClassifierOptions(
      base_options=base_options, classification_options=classification_options)
  detector = vision.ImageClassifier.create_from_options(options)
  imgNameList = ['OkraTest.jpg', 'OkraTest2.jpg', 'OkraTest3.jpg', 'NutgrassTest.jpg', 'NutgrassTest2.jpg', 'NutgrassTest3.jpg']
#  for i in range(6):
#    img = cv2.imread(imgNameList[np.random.randint(6)])
#    resize = tf.image.resize(img,(256,256))
#    resize = resize.astype(np.uint8)
#    #print(resize)
#    input_tensor = vision.TensorImage.create_from_array(resize)
#    classification_result = detector.classify(input_tensor)
#    print(classification_result.classifications[0].categories[0].score)
#    if(classification_result.classifications[0].categories[0].score < 0.5):
#      print("Nutgrass")
#    else:
#      print("Okra")
#    img = cv2.imread(imgNameList[np.random.randint(6)])
#    resize = tf.image.resize(img,(256,256))
#    resize = resize.astype(np.uint8)
#    #print(resize)
#    input_tensor = vision.TensorImage.create_from_array(resize)
#    classification_result = detector.classify(input_tensor)
#    print(classification_result.classifications[0].categories[0].score)
#    if(classification_result.classifications[0].categories[0].score < 0.5):
#      print("Nutgrass")
#    else:
#      print("Okra")
#    img = cv2.imread(imgNameList[np.random.randint(6)])
#    resize = tf.image.resize(img,(256,256))
#    resize = resize.astype(np.uint8)
#    #print(resize)
#    input_tensor = vision.TensorImage.create_from_array(resize)
#    classification_result = detector.classify(input_tensor)
#    print(classification_result.classifications[0].categories[0].score)
#    if(classification_result.classifications[0].categories[0].score < 0.5):
#      print("Nutgrass")
#    else:
#      print("Okra")
  print("Okra 1-3")
  img = cv2.imread('OkraTest.jpg')
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(classification_result)
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra")

  img = cv2.imread('OkraTest2.jpg')
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(classification_result)
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra")

  img = cv2.imread('OkraTest3.jpg')
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(classification_result)
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra")

  print("Nutgrass 1-3")
  img = cv2.imread('NutgrassTest.jpg')
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(classification_result)
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra")

  img = cv2.imread('NutgrassTest2.jpg')
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(classification_result)
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra")

  img = cv2.imread('NutgrassTest3.jpg')
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(classification_result)
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra") 


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the image classification model.',
      required=False,
      default='model_edgetpu_w_MetaData.tflite')
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()