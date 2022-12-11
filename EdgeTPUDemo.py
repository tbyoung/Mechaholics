import argparse
import sys
import time
from tokenize import String
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


def run(model: str, num_threads: int, enable_edgetpu: bool, image: str) -> None:
  start_time = time.time()
  #print('Test Start Time: ', start_time)
  # Initialize the object classification model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  classification_options = processor.ClassificationOptions(
      max_results=2)
  options = vision.ImageClassifierOptions(
      base_options=base_options, classification_options=classification_options)
  detector = vision.ImageClassifier.create_from_options(options)
  start_time = time.time()
  #print('Image Load & Inference Start Time: ', start_time)
  img = cv2.imread(image)
  resize = tf.image.resize(img,(256,256))
  resize = resize.astype(np.uint8)
  #print(resize)
  input_tensor = vision.TensorImage.create_from_array(resize)
  classification_result = detector.classify(input_tensor)
  print(' ')
  print(' ')
  print('Score: ',classification_result.classifications[0].categories[0].score )
  if(classification_result.classifications[0].categories[0].score < 0.5):
    print("Nutgrass")
  else:
    print("Okra")
  inf1_time = time.time()
  #print('Inference End Time: ', inf1_time)
  dif_time = inf1_time - start_time
  print('Time for inference: ', str(dif_time))

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
  parser.add_argument(
      '--image',
      help='Image file',
      required=False,
      default='OkraTest.jpg')    
  args = parser.parse_args()

  run(args.model, int(args.numThreads), bool(args.enableEdgeTPU), str(args.image))


if __name__ == '__main__':
  main()