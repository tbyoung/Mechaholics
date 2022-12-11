import argparse
import time

from PIL import Image
from sys import settrace
import L1_Classify as classify
import tflite_runtime.interpreter as tflite
import platform

import L1_Camera as cam
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

#Variable containing names of test images taken for the AMAR platform
imgNames = [["pigweed_1.jpg", "pigweed_2.jpg", "pigweed_3.jpg", "pigweed_4.jpg", "pigweed_5.jpg", "pigweed_6.jpg", "pigweed_7.jpg", "pigweed_8.jpg", "pigweed_9.jpg", "pigweed_10.jpg", "pigweed_11.jpg", "pigweed_12.jpg"],
  ["turnip_1.jpg", "turnip_2.jpg", "turnip_3.jpg", "turnip_4.jpg", "turnip_5.jpg", "turnip_6.jpg", "turnip_7.jpg", "turnip_8.jpg", "turnip_9.jpg", "turnip_10.jpg", "turnip_11.jpg", "turnip_12.jpg", "img.jpg"]]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

#makes interpreter in "global context", discontinues need for reinitialization of model when running on AMAR
labels = load_labels("amar_labels.txt")

interpreter = make_interpreter("AMAR_Model_Final_quant_edgetpu.tflite")
interpreter.allocate_tensors()

def getClassification():
  start = time.perf_counter()
  cam.imgTake()
  input = "img.jpg"
  image_get = time.perf_counter() - start
  size = classify.input_size(interpreter)
  image = Image.open(input).convert('RGB').resize(size, Image.ANTIALIAS)
  classify.set_input(interpreter, image)

  interpreter.invoke()
  classes = classify.get_output(interpreter, 2, 0)
  interpret = time.perf_counter() - image_get - start
  total_time = time.perf_counter() - start
  print("Image time: ", image_get*1000, "ms Interpret Time: ", interpret*1000, "ms Total Time: ", total_time*1000, "ms")
  inference_result = [[labels.get(classes[0].id, classes[0].id), classes[0].score], [labels.get(classes[1].id, classes[1].id), classes[1].score]] #Only 2 classes for now
  return inference_result

if __name__ == '__main__':
  labels = load_labels("amar_labels.txt")

  interpreter = make_interpreter("AMAR_Model_Final_quant_edgetpu.tflite")
  interpreter.allocate_tensors()
  for n in imgNames[0]:
    size = classify.input_size(interpreter)
    image = Image.open(n).convert('RGB').resize(size, Image.ANTIALIAS)
    classify.set_input(interpreter, image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_output(interpreter, 2, 0)
    print('%.1fms' % (inference_time * 1000))
    for klass in classes:
      print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score), " ", n)
  for n in imgNames[1]:
    size = classify.input_size(interpreter)
    image = Image.open(n).convert('RGB').resize(size, Image.ANTIALIAS)
    classify.set_input(interpreter, image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_output(interpreter, 2, 0)
    print('%.1fms' % (inference_time * 1000))
    for klass in classes:
      print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score), " ", n)