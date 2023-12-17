import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.plugins import projector
from numpy.random import default_rng
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation
from keras.layers import Flatten, Dense
from keras.models import Model

image_dir = "Data/Product Recoginition"
train_dir = os.path.join(image_dir, "Training Data")

all_images = 0
class_list = sorted(os.listdir(train_dir))
for i in class_list:
    all_images += len(os.listdir(os.path.join(train_dir, i)))

_BATCH_SIZE = 32


def plot_triplets(examples):
    fig, axs = plt.subplots(len(examples[0]), 3)
    for i in range(len(examples[0])):
        for j in range(len(examples)):

            axs[j, i].imshow(examples[i][j])
            plt.xticks([])
            plt.yticks([])
    plt.show()


def create_batch(batch_size=_BATCH_SIZE):
    rng = default_rng()

    x_anc = np.zeros((batch_size, 224, 224, 3))
    x_pos = np.zeros((batch_size, 224, 224, 3))
    x_neg = np.zeros((batch_size, 224, 224, 3))

    rand_classes = rng.choice(len(class_list), batch_size, replace=True)

    for i in range(batch_size):
        class_idx = rand_classes[i]
        anc_class_dir = os.path.join(train_dir, class_list[class_idx])
        anc_imglist = os.listdir(anc_class_dir)
        anc_pos_idx = rng.choice(len(anc_imglist), 2, replace=False)

        img_anc = tf.keras.preprocessing.image.load_img(os.path.join(anc_class_dir, anc_imglist[anc_pos_idx[0]]))
        img_anc = img_anc.resize((224, 224))
        img_anc = tf.keras.preprocessing.image.img_to_array(img_anc)

        img_pos = tf.keras.preprocessing.image.load_img(os.path.join(anc_class_dir, anc_imglist[anc_pos_idx[1]]))
        img_pos = img_pos.resize((224, 224))
        img_pos = tf.keras.preprocessing.image.img_to_array(img_pos)

        # Select negative class (!= class_idx)
        all_class_idx = np.arange(len(class_list))
        neg_mask = (all_class_idx != class_idx)

        all_neg_classes = all_class_idx[neg_mask]
        neg_class_idx = rng.choice(all_neg_classes)

        neg_class_dir = os.path.join(train_dir, class_list[neg_class_idx])
        neg_imglist = os.listdir(neg_class_dir)
        neg_idx = rng.choice(len(neg_imglist))

        img_neg = tf.keras.preprocessing.image.load_img(os.path.join(neg_class_dir, neg_imglist[neg_idx]))
        img_neg = img_neg.resize((224, 224))
        img_neg = tf.keras.preprocessing.image.img_to_array(img_neg)

        x_anc[i] = img_anc / 255.
        x_pos[i] = img_pos / 255.
        x_neg[i] = img_neg / 255.

    return [x_anc, x_pos, x_neg]


exp_triplet = create_batch(3)
plot_triplets(exp_triplet)

emb_size = 128

# Build a Small Convnet from scratch

img_input = Input(shape=(224, 224, 3))

# conv layer 1
x = Conv2D(16, 3, padding='same')(img_input)
x = BatchNormalization(axis = 3, name = 'bn0')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2)(x)
# 112 * 112 * 16

# conv layer 2
x = Conv2D(32, 3, padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2)(x)
# 56 * 56 * 32

# conv layer 3
x = Conv2D(64, 3, padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2)(x)
# 28 * 28 * 64

# conv layer 4
x = Conv2D(128, 3, padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2)(x)
# 14 * 14 * 128

x = Flatten()(x)
output = Dense(emb_size, activation='sigmoid')(x)

embedding_model = Model(img_input, output)

input_anchor = tf.keras.layers.Input(shape=(224, 224, 3))
input_positive = tf.keras.layers.Input(shape=(224, 224, 3))
input_negative = tf.keras.layers.Input(shape=(224, 224, 3))

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

all_output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], all_output)


alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


def data_generator(batch_size=256):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y


def pos_mean(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)

    return tf.reduce_mean(positive_dist)


def neg_mean(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)

    return tf.reduce_mean(negative_dist)


_DIST_THRD = 0.1


def metrics_val(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)

    # Count the number of all test data
    all_mask = tf.math.greater_equal(positive_dist, tf.constant([0.]))
    all_num = tf.cast(all_mask, tf.float32)

    positive_dist_led = tf.math.less_equal(positive_dist, tf.constant([_DIST_THRD]))
    positive_dist_led_num = tf.cast(positive_dist_led, tf.float32)

    val = tf.math.divide(positive_dist_led_num, all_num)

    return val


def metrics_far(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :emb_size], y_pred[:, emb_size:2 * emb_size], y_pred[:, 2 * emb_size:]
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    all_mask = tf.math.greater_equal(negative_dist, tf.constant([0.]))
    all_num = tf.cast(all_mask, tf.float32)

    negative_dist_led = tf.math.less_equal(negative_dist, tf.constant([_DIST_THRD]))
    negative_dist_led_num = tf.cast(negative_dist_led, tf.float32)

    far = tf.math.divide(negative_dist_led_num, all_num)

    return far

test_dir = os.path.join(image_dir, "Validation Data")
print(test_dir)
print("items = {}".format(len(os.listdir(test_dir))))

all_images_test = 0
class_list_test = sorted(os.listdir(test_dir))
for i in class_list_test:
    all_images_test += len(os.listdir(os.path.join(test_dir, i)))
print("All test image files = {}".format(all_images_test))

import datetime

log_dir_rel = "logs/logs-{}".format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
log_dir = os.path.abspath(log_dir_rel)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

_TB_METADATA = 'metadata.tsv'


class tbProjector(tf.keras.callbacks.Callback):

    def __init__(self, embedding_model, x_test, y_test, log_dir, metadata):
        super(tbProjector, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.log_dir = log_dir
        self.metadata = metadata
        self.output()

    def output(self):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        test_emb_tensor = tf.Variable(x_test_embeddings)
        checkpoint = tf.train.Checkpoint(embedding=test_emb_tensor)
        checkpoint.save(os.path.join(self.log_dir, "embedding.ckpt"))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = self.metadata
        projector.visualize_embeddings(self.log_dir, config)

    def on_epoch_end(self, epoch, logs=None):
        self.output()


def create_testdata(class_size):
    rng = default_rng()

    test_data_list = []
    test_data_label = []

    all_class_size = len(class_list_test)
    if class_size > all_class_size:
        class_size = all_class_size

    class_list_test_rand = rng.choice(class_list_test, class_size)

    for test_class in class_list_test_rand:

        test_class_dir = os.path.join(test_dir, test_class)
        imglist = os.listdir(test_class_dir)

        for img_file in imglist:
            # Add this class to the list
            test_data_label.append(test_class)

            img = tf.keras.preprocessing.image.load_img(os.path.join(test_class_dir, img_file))
            img = img.resize((224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img /= 255.

            test_data_list.append(img)

    test_data = np.array(test_data_list)

    # Write metadata file
    with open(os.path.join(log_dir, _TB_METADATA), "w") as f:
        for label in test_data_label:
            f.write("{}\n".format(label))

    return test_data


x_test = create_testdata(10)
y_test = np.zeros((x_test.shape[0], emb_size))
print(x_test.shape)
print(y_test.shape)


batch_size = _BATCH_SIZE
epochs = 3
steps_per_epoch = int(all_images/batch_size)
print("step_per_epoch = {}".format(steps_per_epoch))
print("Tensorboard log_dir = {}".format(log_dir))

net.compile(loss=triplet_loss, optimizer='adam', metrics=[
    pos_mean, neg_mean, metrics_val, metrics_far])

_ = net.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[
        tbProjector(
            embedding_model,
            x_test, y_test,
            log_dir, _TB_METADATA
        )]
)


