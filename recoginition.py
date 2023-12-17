import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, Activation, Flatten, Dense
from keras.models import Model
from tensorboard.plugins import projector
import datetime
from sklearn.metrics import accuracy_score


# Define constants
base_dir = os.getcwd()
_IMAGE_DIR = os.path.join(base_dir, "Data/Product Recoginition")
_TRAIN_DIR = os.path.join(_IMAGE_DIR, "Training Data")
_VAL_DIR = os.path.join(_IMAGE_DIR, "Validation Data")
_BATCH_SIZE = 64
_EMB_SIZE = 128
_ALPHA = 0.2
_DIST_THRD = 0.1
_LOG_DIR_REL = "logs/logs-{}".format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
_TB_METADATA = 'metadata.tsv'
_TEST_DIR = "Data/Test Classification"
_EPOCHS = 4
log_dir_rel = "logs/logs-{}".format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
log_dir = os.path.abspath(log_dir_rel)
all_images_val = 0
all_images = 0
class_list = sorted(os.listdir(_TRAIN_DIR))
class_list_val = sorted(os.listdir(_VAL_DIR))
class_list_test = sorted(os.listdir(_TEST_DIR))
for i in class_list_val:
    all_images_val += len(os.listdir(os.path.join(_VAL_DIR, i)))
for i in class_list:
    all_images += len(os.listdir(os.path.join(_TRAIN_DIR, i)))


# Function to create a batch of triplets
def create_batch(List_DIR, class_list, batch_size=_BATCH_SIZE):
    rng = default_rng()
    x_anc = np.zeros((batch_size, 224, 224, 3))
    x_pos = np.zeros((batch_size, 224, 224, 3))
    x_neg = np.zeros((batch_size, 224, 224, 3))
    rand_classes = rng.choice(len(class_list), batch_size, replace=True)

    for i in range(batch_size):
        class_idx = rand_classes[i]
        anc_class_dir = os.path.join(List_DIR, class_list[class_idx])
        anc_imglist = os.listdir(anc_class_dir)
        anc_pos_idx = rng.choice(len(anc_imglist), 2, replace=False)

        img_anc = preprocess_image(os.path.join(anc_class_dir, anc_imglist[anc_pos_idx[0]]))
        img_pos = preprocess_image(os.path.join(anc_class_dir, anc_imglist[anc_pos_idx[1]]))

        all_class_idx = np.arange(len(class_list))
        neg_mask = (all_class_idx != class_idx)
        all_neg_classes = all_class_idx[neg_mask]
        neg_class_idx = rng.choice(all_neg_classes)
        neg_class_dir = os.path.join(List_DIR, class_list[neg_class_idx])
        neg_imglist = os.listdir(neg_class_dir)
        neg_idx = rng.choice(len(neg_imglist))
        img_neg = preprocess_image(os.path.join(neg_class_dir, neg_imglist[neg_idx]))

        x_anc[i] = img_anc
        x_pos[i] = img_pos
        x_neg[i] = img_neg

    return [x_anc, x_pos, x_neg]


# Function to preprocess an image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img /= 255.
    return img


# Function to plot triplets
def plot_triplets(examples):
    fig, axs = plt.subplots(len(examples[0]), 3)
    for i in range(len(examples[0])):
        for j in range(len(examples)):
            axs[j, i].imshow(examples[i][j])
            plt.xticks([])
            plt.yticks([])
    plt.show()

# Model architecture function
def build_embedding_model():
    img_input = Input(shape=(224, 224, 3))
    x = Conv2D(16, 3, padding='same')(img_input)
    x = BatchNormalization(axis=3, name='bn0')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)
    # ... (add more layers as needed)
    x = Flatten()(x)
    output = Dense(_EMB_SIZE, activation='sigmoid')(x)
    return Model(img_input, output)

# Siamese network model function
def build_siamese_model(embedding_model):
    input_anchor = Input(shape=(224, 224, 3))
    input_positive = Input(shape=(224, 224, 3))
    input_negative = Input(shape=(224, 224, 3))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    all_output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    return Model([input_anchor, input_positive, input_negative], all_output)

# Loss function
def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :_EMB_SIZE], y_pred[:, _EMB_SIZE:2*_EMB_SIZE], y_pred[:, 2*_EMB_SIZE:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + _ALPHA, 0.)

# Metrics functions
def pos_mean(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :_EMB_SIZE], y_pred[:, _EMB_SIZE:2*_EMB_SIZE], y_pred[:, 2*_EMB_SIZE:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    return tf.reduce_mean(positive_dist)

def neg_mean(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :_EMB_SIZE], y_pred[:, _EMB_SIZE:2*_EMB_SIZE], y_pred[:, 2*_EMB_SIZE:]
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(negative_dist)

def metrics_val(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :_EMB_SIZE], y_pred[:, _EMB_SIZE:2*_EMB_SIZE], y_pred[:, 2*_EMB_SIZE:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    all_mask = tf.math.greater_equal(positive_dist, tf.constant([0.]))
    all_num = tf.cast(all_mask, tf.float32)
    positive_dist_led = tf.math.less_equal(positive_dist, tf.constant([_DIST_THRD]))
    positive_dist_led_num = tf.cast(positive_dist_led, tf.float32)
    val = tf.math.divide(positive_dist_led_num, all_num)
    return val

def metrics_far(y_true, y_pred):
    anchor, positive, negative = y_pred[:, :_EMB_SIZE], y_pred[:, _EMB_SIZE:2*_EMB_SIZE], y_pred[:, 2*_EMB_SIZE:]
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    all_mask = tf.math.greater_equal(negative_dist, tf.constant([0.]))
    all_num = tf.cast(all_mask, tf.float32)
    negative_dist_led = tf.math.less_equal(negative_dist, tf.constant([_DIST_THRD]))
    negative_dist_led_num = tf.cast(negative_dist_led, tf.float32)
    far = tf.math.divide(negative_dist_led_num, all_num)
    return far

# Function to create the test data
def create_testdata(class_size):
    rng = default_rng()
    test_data_list = []
    test_data_label = []
    all_class_size = len(class_list_test)

    class_list_test_rand = rng.choice(class_list_test, min(class_size, all_class_size), replace=False)

    for test_class in class_list_test_rand:
        test_class_dir = os.path.join(_TEST_DIR, test_class)
        imglist = os.listdir(test_class_dir)

        for img_file in imglist:
            test_data_label.append(test_class)
            img = preprocess_image(os.path.join(test_class_dir, img_file))
            test_data_list.append(img)

    test_data = np.array(test_data_list)

    return test_data


# Callback for TensorBoard projector
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

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = self.metadata
        projector.visualize_embeddings(self.log_dir, config)

    def on_epoch_end(self, epoch, logs=None):
        self.output()


def data_generator(_DIR, class_list, batch_size=256):
    while True:
        x = create_batch(_DIR, class_list, batch_size)
        y = np.zeros((batch_size, 3 * _EMB_SIZE))
        yield x, y


# Main training code
def train_siamese_network():
    embedding_model = build_embedding_model()
    siamese_model = build_siamese_model(embedding_model)

    x_test = create_testdata(10)  # Use a small subset of the training data for testing
    y_test = np.zeros((x_test.shape[0], _EMB_SIZE))

    batch_size = _BATCH_SIZE
    steps_per_epoch_train = int(all_images / batch_size)
    steps_per_epoch_val = int(all_images_val / batch_size)

    # Separate the data generators for training and validation
    train_data_gen = data_generator(_TRAIN_DIR, class_list, batch_size)
    val_data_gen = data_generator(_VAL_DIR, class_list_val, batch_size)

    siamese_model.compile(loss=triplet_loss, optimizer='adam', metrics=[pos_mean, neg_mean, metrics_val, metrics_far])

    siamese_model.fit(
        train_data_gen,
        steps_per_epoch=steps_per_epoch_train,
        epochs=_EPOCHS,
        validation_data=val_data_gen,
        validation_steps=steps_per_epoch_val,
        callbacks=[tbProjector(embedding_model, x_test, y_test, log_dir, _TB_METADATA)]
    )
    return siamese_model


# Run the training
exp_triplet = create_batch(_TRAIN_DIR, class_list,3)
plot_triplets(exp_triplet)
SIAMESE_MODEL = train_siamese_network()


# Scenario 1
def find_most_similar_image(siamese_model, anchor_path, folder_path):
    anchor_img = preprocess_image(anchor_path)
    anchor_embedding = siamese_model.layers[2](anchor_img)

    most_similar_image = None
    min_distance = float('inf')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            test_img = preprocess_image(image_path)
            test_embedding = siamese_model.layers[2](test_img)

            distance = tf.norm(anchor_embedding - test_embedding).numpy()

            if distance < min_distance:
                min_distance = distance
                most_similar_image = filename

    return most_similar_image


anchor_image_path = "Data/Test Classification/1/web15.jpg"
folder_path = "Data/test_scenario1"
most_similar_image = find_most_similar_image(SIAMESE_MODEL, anchor_image_path, folder_path)
print(f"The most similar image to the anchor is: {most_similar_image}")


# Senario 2
def generate_embeddings(siamese_model, image_path):
    img = preprocess_image(image_path)
    embedding = siamese_model.layers[2](img)
    return embedding.numpy()

def predict_product_id(siamese_model, test_image_path, training_folders):
    test_embedding = generate_embeddings(siamese_model, test_image_path)
    #means = []
    distances = []
    for training_folder in training_folders:
        product_id = os.path.basename(training_folder)
        for filename in os.listdir(training_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                training_img_path = os.path.join(training_folder, filename)
                training_embedding = generate_embeddings(siamese_model, training_img_path)

                distance = tf.norm(test_embedding - training_embedding).numpy()
                distances.append([product_id, distance])
        #distance_mean = np.mean(distances)
        #means.append([product_id, distance_mean])

    min_distance = distances[0][1]
    predicted_product_id =  distances[0][0]
    for i in range(len(distances)):
        if distances[i][1] < min_distance:
            min_distance = distances[i][1]
            predicted_product_id = distances[i][0]
    return predicted_product_id

def evaluate(siamese_model, test_folder, training_folders):
    true_labels = []
    predicted_labels = []

    for class_folder in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_folder)
        classe_label = os.path.basename(class_folder)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_img_path = os.path.join(class_path, filename)

                true_labels.append(classe_label)

                predicted_product_id = predict_product_id(siamese_model, test_img_path, training_folders)
                predicted_labels.append(predicted_product_id)
                print(f"Test Image: {filename}, Test Class: {classe_label}, Predicted Product ID: {predicted_product_id}")

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy


training_folders = [os.path.join(_TRAIN_DIR, folder) for folder in os.listdir(_TRAIN_DIR)]
test_dir = os.path.relpath("Data/test_scenario_2")
accuracy_test = evaluate(SIAMESE_MODEL, test_dir, training_folders)

print(f"\nAccuracy on Test Set: {accuracy_test}")

