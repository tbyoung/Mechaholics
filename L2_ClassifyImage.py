import argparse
import time

from PIL import Image
from sys import settrace
import L1_Classify
import tflite_runtime.interpreter as tflite
import platform

import L1_Camera as cam
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


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
labels = load_labels("plant_labels.txt")

interpreter = make_interpreter("AMAR_Model_quant_W_NoPlant_edgetpu.tflite")
interpreter.allocate_tensors()

def getClassification():
  start = time.perf_counter()
  cam.imgTake()
  input = "img.jpg"
  image_get = time.perf_counter() - start
  size = L1_Classify.input_size(interpreter)
  image = Image.open(input).convert('RGB').resize(size, Image.ANTIALIAS)
  L1_Classify.set_input(interpreter, image)

  interpreter.invoke()
  classes = L1_Classify.get_output(interpreter, 2, 0)
  interpret = time.perf_counter() - image_get - start
  total_time = time.perf_counter() - start
  print("Image time: ", image_get*1000, "ms Interpret Time: ", interpret*1000, "ms Total Time: ", total_time*1000, "ms")
  inference_result = [[labels.get(classes[0].id, classes[0].id), classes[0].score], [labels.get(classes[1].id, classes[1].id), classes[1].score]] #Only 2 classes for now
  return inference_result

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  size = L1_Classify.input_size(interpreter)
  image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
  L1_Classify.set_input(interpreter, image)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = L1_Classify.get_output(interpreter, args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  for klass in classes:
    print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))