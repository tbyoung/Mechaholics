# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to classify a given image using an Edge TPU.
To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.
Example usage:
```
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import argparse
import time

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

imgNames = [["pigweed_1.jpg", "pigweed_2.jpg", "pigweed_3.jpg", "pigweed_4.jpg", "pigweed_5.jpg", "pigweed_6.jpg", "pigweed_7.jpg", "pigweed_8.jpg", "pigweed_9.jpg", "pigweed_10.jpg", "pigweed_11.jpg", "pigweed_12.jpg"],
  ["turnip_1.jpg", "turnip_2.jpg", "turnip_3.jpg", "turnip_4.jpg", "turnip_5.jpg", "turnip_6.jpg", "turnip_7.jpg", "turnip_8.jpg", "turnip_9.jpg", "turnip_10.jpg", "turnip_11.jpg", "turnip_12.jpg"]]

def main():
#  parser = argparse.ArgumentParser(
#      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#  parser.add_argument(
#      '-m', '--model', required=True, help='File path of .tflite file.')
#  parser.add_argument(
#      '-i', '--input', required=True, help='Image to be classified.')
#  parser.add_argument(
#      '-l', '--labels', help='File path of labels file.')
#  parser.add_argument(
#      '-k', '--top_k', type=int, default=1,
#      help='Max number of classification results')
#  parser.add_argument(
#      '-t', '--threshold', type=float, default=0.0,
#      help='Classification score threshold')
#  parser.add_argument(
#      '-c', '--count', type=int, default=5,
#      help='Number of times to run inference')
#  parser.add_argument(
#      '-a', '--input_mean', type=float, default=128.0,
#      help='Mean value for input normalization')
#  parser.add_argument(
#      '-s', '--input_std', type=float, default=128.0,
#      help='STD value for input normalization')
#  args = parser.parse_args()

  labels = read_label_file("plant_labels.txt")
  print("read labels")
  interpreter = make_interpreter("AMAR_Model_Final_quant_edgetpu.tflite")
  interpreter.allocate_tensors()
  # Model must be uint8 quantized
  if common.input_details(interpreter, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')

  size = common.input_size(interpreter)

  for n in imgNames[0]:
    image = Image.open(n).convert('RGB').resize(size, Image.ANTIALIAS)

    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean = 128
    std = 128
    if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
    # Input data does not require preprocessing.
      common.set_input(interpreter, image)
    else:
      # Input data requires preprocessing
      normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
      np.clip(normalized_input, 0, 255, out=normalized_input)
      common.set_input(interpreter, normalized_input.astype(np.uint8))
    # Run inference
    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(1):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      classes = classify.get_classes(interpreter, 2, 0)
      print('%.1fms' % (inference_time * 1000))
      for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score), " ", n)

  for n in imgNames[1]:
    image = Image.open(n).convert('RGB').resize(size, Image.ANTIALIAS)

    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean = 128
    std = 128
    if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
    # Input data does not require preprocessing.
      common.set_input(interpreter, image)
    else:
      # Input data requires preprocessing
      normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
      np.clip(normalized_input, 0, 255, out=normalized_input)
      common.set_input(interpreter, normalized_input.astype(np.uint8))
    # Run inference
    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(1):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      classes = classify.get_classes(interpreter, 2, 0)
      print('%.1fms' % (inference_time * 1000))
      for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score), " ", n)


if __name__ == '__main__':
  main()
else:
  labels = read_label_file("plant_labels.txt")
  print("read labels")
  interpreter = make_interpreter("AMAR_Model_Final_quant_rev2_edgetpu.tflite")
  interpreter.allocate_tensors()

def getClassification():
  print("made into function")
  size = common.input_size(interpreter)
  print("set size")
  image = Image.open("img.jpg").convert('RGB').resize(size, Image.ANTIALIAS)
  print("opened image")
  # Image data must go through two transforms before running inference:
  # 1. normalization: f = (input - mean) / std
  # 2. quantization: q = f / scale + zero_point
  # The following code combines the two steps as such:
  # q = (input - mean) / (std * scale) + zero_point
  # However, if std * scale equals 1, and mean - zero_point equals 0, the input
  # does not need any preprocessing (but in practice, even if the results are
  # very close to 1 and 0, it is probably okay to skip preprocessing for better
  # efficiency; we use 1e-5 below instead of absolute zero).
  params = common.input_details(interpreter, 'quantization_parameters')
  scale = params['scales']
  zero_point = params['zero_points']
  mean = 128
  std = 128
  if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
    # Input data does not require preprocessing.
    common.set_input(interpreter, image)
  else:
    # Input data requires preprocessing
    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    common.set_input(interpreter, normalized_input.astype(np.uint8))

  # Run inference
  #print('----INFERENCE TIME----')
  #print('Note: The first inference on Edge TPU is slow because it includes',
  #      'loading the model into Edge TPU memory.')
  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  classes = classify.get_classes(interpreter, 2, 0) #argument is - get_classes(interpreter, top x classes, threshold is x score)
  print('%.1fms' % (inference_time * 1000))
  return [[labels.get(classes[0].id,classes[0].id), classes[0].score], [labels.get(classes[1].id, classes[1].id), classes[1].score]] 
