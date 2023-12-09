import os
import numpy as np
import tensorflow as tf
import keras
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# keras.applications.vgg16.VGG16
model_vgg16 = VGG16(weights='imagenet', include_top=False)
print(model_vgg16.summary())
print(model_vgg16.layers)  # 19

def extract_features(images):
    features = []
    for img_path in images:
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model_vgg16.predict(x).flatten())
    return np.array(features)

num_classes = 20
train_images = []
train_labels = []
validation_images = []
validation_labels = []

for i in range(1, num_classes + 1):
    train_dir = "Product Classification/" + str(i) + "/Train"
    validation_dir = "Product Classification/" + str(i) + "/Validation"
    class_name = str(i)

    for filename in os.listdir(train_dir):
        if filename.endswith('.png'):
            img_path1 = os.path.join(train_dir, filename)
            train_images.append(img_path1)
            train_labels.append(class_name)

    for filename in os.listdir(validation_dir):
        if filename.endswith('.png'):
            img_path2 = os.path.join(validation_dir, filename)
            validation_images.append(img_path2)
            validation_labels.append(class_name)
#
# print("////", len(train_images))  # 140
# print("#####", len(validation_images))  # 34
train_features = extract_features(train_images)
validation_features = extract_features(validation_images)

logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(train_features, train_labels)
validation_predictions = logistic_regression.predict(validation_features)
print(validation_predictions)
print(validation_labels)
train_predictions = logistic_regression.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
print("Accuracy Logistic Regression on train data:", train_accuracy)
accuracy = accuracy_score(validation_labels, validation_predictions)
print("Accuracy Logistic Regression on validation data:", accuracy)
#  train data: 1.0
#  validation data: 0.9705882352941176

#                                       10                       16   17                                               27
# ['1' '1' '1' '2' '2' '3' '3' '4' '4' '5'  '5' '7' '7' '8' '8' '17' '9' '10' '10' '11' '11' '12' '12' '13' '13' '14' '14' '15' '15' '16' '17' '18' '19' '20']
# ['1' '1' '1' '2' '2' '3' '3' '4' '4' '5'  '5' '7' '7' '8' '8' '9'  '9' '10' '10' '11' '11' '12' '12' '13' '13' '14' '14' '15' '15' '16' '17' '18' '19' '20']
# ['1' '1' '1' '2' '2' '3' '3' '4' '4' '15' '5' '7' '7' '8' '8' '15' '17''10' '10' '11' '11' '12' '12' '13' '13' '14' '3'  '15' '15' '16' '17' '18' '19' '20']



svm_model = SVC(kernel='linear', C=0.1, probability=True)
svm_model.fit(train_features, train_labels)
train_predictions2 = svm_model.predict(train_features)
train_accuracy2 = accuracy_score(train_labels, train_predictions2)
print("Accuracy SVM on train data:", train_accuracy2)
svm_predictions = svm_model.predict(validation_features)
print("SVM Predictions:", svm_predictions)
svm_accuracy = svm_model.score(validation_features, validation_labels)
print("Accuracy SVM on validation data:", svm_accuracy)
# # train data: 1.0
# # validation data: 0.8823529411764706a

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
