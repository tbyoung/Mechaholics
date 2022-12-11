import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3 #Tensorflow version must be higher than 2.3 for this Transfer Learning to work

import os
import numpy as np
import matplotlib.pyplot as plt

plant_dir = os.path.join('data') #Directory that contains all images for use with transfer learning
IMAGE_SIZE = 224 #Image size for Mobilenet v2
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator( #Create datagenerator for model trainging and evaluation
    rescale=1./255, #Rescale image for variables from (0-255) to (0-1) for faster/easier training (less hard math)
    validation_split=0.2)

train_generator = datagen.flow_from_directory( #Generate data for training the model
    plant_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory( #Generate data for evaluating the model
    plant_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

image_batch, label_batch = next(val_generator) #Create a batch for the evaluation data
image_batch.shape, label_batch.shape
print (train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys())) #create label data

with open('plant_labels.txt', 'w') as f: #Save label data as txt (easy to read and open in any OS, easy to modify in case of necessary changes)
  f.write(labels)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3) #Define image shape for model

# Create the base model from the pre-trained MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,  #Utilizing the MobileNetV2 base model (and requisite specifications); beginning of transfer learning
                                              include_top=False, 
                                              weights='imagenet')
base_model.trainable = False #do not change current base model for now
model = tf.keras.Sequential([ #Our modifications to the base model
  base_model, #Base model on top of our modifications
  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'), #Conv2D layer - applies 32 filters (to extract feature map) with 3 kernel size (size of filter applied across image)
  tf.keras.layers.Dropout(0.2), #Dropout layer, prevents overfitting of model even if we train for too many epochs
  tf.keras.layers.GlobalAveragePooling2D(), #Pooling layer, accepts 4D tensor (result from conv2d and dropout) and outputs 2D tensor (batch dimensions, number of channels)
  tf.keras.layers.Dense(units=2, activation='softmax') #Condenses output to 2 tensors specifying class and score with score being how likely an image is to be the associated label based upon the training
])
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()
print('Number of trainable weights = {}'.format(len(model.trainable_weights)))
history = model.fit(train_generator,  #Begin 1st round of training
                    steps_per_epoch=len(train_generator), 
                    epochs=20,
                    validation_data=val_generator,
                    validation_steps=len(val_generator))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print("Number of layers in the base model: ", len(base_model.layers))
base_model.trainable = True #Now we can fine tune the base-model (could have done previously, but this is the setup i came across online)
fine_tune_at = 100 #Originally meant for MobilenetV1 model (which has significantly more layers), however I find that this fine-tune point works for this transfer learning model with mobilenetV2 as well

# Freeze (make untrainable) all the layers before the `fine_tune_at` layer (layer to begin modifying the base model)
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), #This is compiling the model (with previous changes); still with the Adam optomizer; however with a much lower error tolerance
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print('Number of trainable weights = {}'.format(len(model.trainable_weights)))
history_fine = model.fit(train_generator, #retrain the model given the adjustments made
                         steps_per_epoch=len(train_generator), 
                         epochs=20,
                         validation_data=val_generator,
                         validation_steps=len(val_generator))
acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('AMAR_Model_Final.tflite', 'wb') as f: #Save model pre-quantization
  f.write(tflite_model)

# A generator that provides a representative dataset
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(plant_dir + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any operations can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('AMAR_Model_Final_quant.tflite', 'wb') as f: #model for use to convert to edge tpu model. NOTE: This model must still be converted to an edge-tpu model. It IS NOT ready as-is for use on a microcontroller utilizing an Edge TPU!
  f.write(tflite_model)


#NOTE: All code after this is for evaluation of the finished, quantized model and does not matter if all you are looking for is the final model file.
batch_images, batch_labels = next(val_generator)

logits = model(batch_images)
prediction = np.argmax(logits, axis=1)
truth = np.argmax(batch_labels, axis=1)

keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(prediction, truth)

print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))
def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #   input_tensor[:, :] = input
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = np.uint8(input / scale + zero_point)

def classify_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  top_1 = np.argmax(output)
  return top_1

interpreter = tf.lite.Interpreter('mobilenet_v2_1.0_224_quant.tflite')
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)
print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))
