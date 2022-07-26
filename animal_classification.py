import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorboard
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import webbrowser
import datetime
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def parse_single_image(image, label):
    data = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'raw_image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def write_images_to_tfr_short(images, labels, filename:str="images"):
  filename= filename+".tfrecords"
  writer = tf.io.TFRecordWriter(filename)
  count = 0

  for index in range(len(images)):
    current_image = images[index]
    current_label = labels[index]

    out = parse_single_image(image=current_image, label=current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count

# download data from a folder
path_train = os.path.join("animals_resized", "animals_resized")
# categories with animal breeds
categories = ["aby", "am_bul","basset"]
# print(categories)

data_train=[]
label_train =[]
data_train_tfr=[]
label_train_tfr=[]


# downloading training and test data and assigning them a label
for category in categories:
    path1 = os.path.join(path_train, category + "_resized")
    class_num = categories.index(category)

    for img in os.listdir(path1):
        # save the picture to img_trainig
        # to change to greyscale --> img_array_trainig = cv2.imread(os.path.join(path1, img), cv2.IMREAD_GRAYSCALE)
        img_trainig = cv2.imread(os.path.join(path1, img))
        # to resize images
        # img_array_trainig = cv2.resize(img_array_trainig,(img_size,img_size))
        # change to np.array
        img_array_trainig =np.array(img_trainig)
        # colors in a scale from 0 to 1
        img_array_trainig = img_array_trainig/255
        # adding data to the list
        data_train.append(img_array_trainig)
        label_train.append(class_num)
        data_train_tfr.append(img_trainig)
        label_train_tfr.append(class_num)


# replacing the list of data and labels with np.array so that the model can process the data
data_train = np.array(data_train)
label_train = np.array(label_train)


# scrambling data so that they are not sequentially / possible to do this in model invocation
data_testing_shuffled,label_testing_shuffled = shuffle(data_train, label_train, random_state=0)

# division of data into training (train), validation (eval) and test (test)
train_data , x , train_label , y = train_test_split(data_testing_shuffled, label_testing_shuffled ,
                                            test_size = 0.2 ,
                                            random_state = 15)

eval_data , test_data , eval_labels , test_labels = train_test_split(x,y,
                                                                     test_size = 0.5,
                                                                     random_state = 15)

#export training data to tfrecord file
count = write_images_to_tfr_short(data_train_tfr,label_train_tfr, filename="zwierzeta_trening_images")

# checking data format

# print("Training data format:")
# print(train_data .shape)
# print(len(train_label ))
# print(train_label)
#
# print("Validation data format:")
# print(eval_data .shape)
# print(len(eval_labels))
# print(eval_labels)
#
# print("Test data format:")
# print(test_data .shape)
# print(len(test_labels))
# print(test_labels)
#
# # displaying some training data to check that it is in the correct format
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_data[i], cmap=plt.cm.binary)
#     # the values in the label are floats so you have to change it to int
#     j = int(train_label[i])
#     # print(label_train_shuffled[i])
#     # print(j)
#     plt.xlabel(categories[j])
# plt.show()

#  model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = (164,164,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu' ))
model.add(tf.keras.layers.Dense(3,activation='softmax' ))

# print(model.summary())

# model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# # # tensorboar
# # Clear any logs from previous runs
# # rm -rf ./logs/ -- linuks

path_log_dir = "logs/fit/"
if os.path.exists(path_log_dir):
    # removing the folder
    if not shutil.rmtree(path_log_dir):
        # success message
        print(f"{path_log_dir} is removed successfully")
    else:
        # failure message
        print(f"Unable to delete the {path_log_dir}")

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# # Launch TensorBoard via the command line
# # tensorboard --logdir logs/fit
#
# training the model
#  with tensorboard
model.fit(train_data,train_label, epochs=5,
          batch_size=16, validation_data=(eval_data, eval_labels),
          callbacks=[tensorboard_callback])

# os.system("tensorboard --logdir logs/fit")
webbrowser.open('http://localhost:6006/')

# without tensorboard
# model.fit(train_data,train_label, epochs=6,
#           batch_size=16, validation_data=(eval_data,eval_labels))

# without tensorade and without validation data
# model.fit(data_train_shuffled,label_train_shuffled , epochs=5, batch_size = 16)

# check the model separately on the test set
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# predicting - testing the script on an external image provided from the command line
path_test = input("Enter the image location to be checked: ")
img_testowe = cv2.imread(path_test)
# adjusting the size and format of the data
img_testowe = cv2.resize(img_testowe,(164,164))
img_testowe = np.array(img_testowe)
img_testowe = img_testowe/255
img_testowe = img_testowe.reshape(-1, 164,164, 3)
# print(img_testowe.shape)

predictions = model.predict(img_testowe)
x = np.argmax(predictions[0])
print("The animal in the picture is: " + categories[x])

# Export the model to h5 and json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
