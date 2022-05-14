from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import numpy as np
import os
import pickle
import time
import datetime
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
class MyAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="Accuracy", **kwargs):
        super(MyAccuracy,self).__init__(name=name, **kwargs)


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true,1)
        y_pred = tf.argmax(y_pred,1)
        t= tf.reduce_mean()

def mymetrics(y_true, y_pred):
    y_true = tf.argmax(y_true, 1)
    y_pred = tf.argmax(y_pred, 1)
    t = tf.reduce_mean(tf.cast(tf.equal(y_true,y_pred), tf.float32))

    return t
"""
get_image
"""
image_path = 'cifar-100-python'
class_path = os.path.join(image_path,"meta")
train_path =  os.path.join(image_path,"train")
test_path = os.path.join(image_path, "test")

with open(class_path,"rb") as f:
    dic_class = pickle.load(f, encoding='bytes')
    labels = dic_class[b'fine_label_names']

with open(train_path,"rb") as f:
    dic_train = pickle.load(f, encoding='bytes')
    labeled_train = list(zip(dic_train[b'data'], tf.one_hot(indices=dic_train[b'fine_labels'], depth=100, on_value=1.0,
                                                           off_value=0.0)))

rdata = []
rlabel = []
for d,l in labeled_train:
    rdata.append(np.reshape(np.reshape(d,[3,1024]).T, [32,32,3]))
    rlabel.append(l)

rdata = tf.constant(np.asarray(rdata), dtype=tf.float32)
rlabel = tf.constant(np.asarray(rlabel))

with open(test_path, "rb") as f:
    dic_test = pickle.load(f, encoding="bytes")
    data = tf.constant(np.asarray(dic_test[b'data']), dtype=tf.float32)
    test = []
    for d in data:
        test.append((np.reshape(np.reshape(d, [3,1024]).T, [32,32,3])))
    labeled_test = list(zip(test, tf.one_hot(indices=dic_test[b'fine_labels'], depth=100, on_value=1.0,
                                                            off_value=0.0)))
vdata = []
vlabel = []
for d,l in labeled_test:
    vdata.append(d)
    vlabel.append(l)

vdata = tf.constant(np.asarray(vdata), dtype=tf.float32)
vlabel = tf.constant(np.asarray(vlabel))
validation = (vdata, vlabel)

def visualization(img_cutout, img_cutmix, img_mixup):
    batch_size = img_cutout.shape[0]
    samples = random.choices(range(batch_size), k=3)
    plt.figure(figsize=(3,3))
    for i in range(9):
        if i//3 == 0:
            func = 'cutout'
            img = img_cutout[samples[i % 3]]
        elif i//3 == 1:
            func = 'cutmix'
            img = img_cutmix[samples[i % 3]]
        else:
            func = 'mixup'
            img = img_mixup[samples[i % 3]]
        ax = plt.subplot(3, 3, i+1)
        ax.set_title(rlabel[samples[i % 3]])
        plt.imshow(img.numpy().astype("int32"))
        plt.title(f"picture{i%3}:{func}")
        plt.axis("off")
    plt.show()

def cutout(img_data):
    return tfa.image.random_cutout(img_data, mask_size=(4,4), constant_values=0)


def cutmix(image, label, PROBABILITY=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    DIM = image.shape[1]
    AUG_BATCH = image.shape[0]
    imgs = []
    labels = []
    for j in range(AUG_BATCH):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        WIDTH = tf.cast(np.random.beta(0.2, 0.2, size=[]) * DIM, tf.int32) * P
        # WIDTH  = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)*P  # this is beta dist with alpha=1.0
        # WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        labels.append(a * label[k] + (1 - a) * label[j])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
    return image2, labels


def mixup(image, label, PROBABILITY=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = image.shape[1]
    AUG_BATCH = image.shape[0]
    imgs = []
    labels = []
    for j in range(AUG_BATCH):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
        a = np.random.beta(0.2, 0.2, size=[])   # this is beta dist with alpha=0.2
        # MAKE MIXUP IMAGE
        img1 = image[j, ]
        img2 = image[k, ]
        imgs.append(a * img1 + (1-a) * img2)
        labels.append(a*label[j] + (1-a)*label[k])

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
    image2 = tf.cast(image2, tf.int16)
    return image2, labels


mixup_img, mixup_label = mixup(rdata, rlabel)
mixup_label = tf.constant(np.asarray(mixup_label))

cutmix_img, cutmix_label = cutmix(rdata, rlabel)
cutmix_label = tf.constant(np.asarray(cutmix_label))

cutout_img = cutout(rdata)
val_cutout = cutout(vdata)
#
# processed_img = {'img_mixup':mixup_img, "img_cutout":cutout_img, "img_cutmix":cutmix_img}
#
# visualization(**processed_img)

data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1),]
)



base_model = ResNet50(
    include_top=False,
    input_shape=(32, 32, 3),
    weights='imagenet')

base_model.trainable = False
inputs = keras.Input(shape=(32,32,3))

x = data_augmentation(inputs)
# x = inputs
# scale_layer = keras.layers.Rescaling(scale=1/127.5, offset=-1)
# x = scale_layer(x)

x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(500, activation="relu")(x)
x = keras.layers.Dropout(0.05)(x)
outputs = keras.layers.Dense(100)(x)
model = keras.Model(inputs, outputs)

base_model.trainable = False
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = 1e-3
epochs = 30
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=tf.compat.v1.train.cosine_decay(global_step=global_step, learning_rate=learning_rate, decay_steps=1667*epochs)),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=100,
                                                      mode='min')

"""
Baseline
"""
log_dir = "logs/fit/" + "baseline" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
baseline_history = model.fit(x=rdata, y=rlabel, epochs=epochs,batch_size=60, callbacks=[tensorboard_callback], validation_data=validation)
loss, accuracy = model.evaluate(x=vdata, y=vlabel)
print("Baseline: loss={}, accuracy={}".format(loss, accuracy))
model_path = r"Models/baseline_model"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(model_path, include_optimizer=False)

"""
CutMix
"""
log_dir = "logs/fit/" + "cutmix"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cutmix_history = model.fit(x=cutmix_img, y=cutmix_label, epochs=epochs,batch_size=60, callbacks=[tensorboard_callback], validation_data=validation)
loss, accuracy = model.evaluate(x=vdata, y=vlabel)
print("CutMix: loss={}, accuracy={}".format(loss, accuracy))
model_path = r"Models/cutmix_model_finetune"
model.save(model_path, include_optimizer=False)

"35.35"

"""
Cutout
"""
log_dir = "logs/fit/" + "cutout"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cutout_history = model.fit(x=cutout_img, y=rlabel, epochs=epochs,batch_size=60, callbacks=[tensorboard_callback], validation_data=validation)
loss, accuracy = model.evaluate(x=vdata, y=vlabel)
print("Cutout: loss={}, accuracy={}".format(loss, accuracy))
model_path = r"Models/cutout_model"
model.save(model_path, include_optimizer=False)

"36.25"

"""
Mixup
"""
log_dir = "logs/fit/" + "mixup"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
mixup_history = model.fit(x=mixup_img, y=mixup_label, epochs=epochs,batch_size=60, callbacks=[tensorboard_callback], validation_data=validation)
loss, accuracy = model.evaluate(x=vdata, y=vlabel)
print("Mixup: loss={}, accuracy={}".format(loss, accuracy))
model_path = r"Models/mixup_model"
model.save(model_path, include_optimizer=False)

"36.3"


def predict_image(image_path,model_path):
    """
    Notice that the image size should be 32*32*3
    """
    image = plt.imread(image_path)
    model = tf.keras.models.load_model(model_path)
    plt.imshow(image)
    image = tf.expand_dims(image, 0)
    # image = plt.imread(image_path)
    res = int(tf.argmax(model(image), 1))
    plt.title(str(labels[res]))
    plt.show()
