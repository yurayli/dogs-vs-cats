# load libraries
from data_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, Flatten, Activation, concatenate
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.applications.resnet50 import identity_block, conv_block


def log_loss(y_true, y_pred):
    y_pred = np.maximum(np.minimum(y_pred, 1 - 1e-6), 1e-6)
    return -np.mean([np.log(p) if y_true[i]==1 else np.log(1-p) for i,p in enumerate(y_pred)])

def predict(test_path, model):
    ims, inps = load_image_label(test_path, label=False)
    ims = np.array([im.split('.')[0] for im in ims])
    preds = model.predict(inps, batch_size=64, verbose=1)[:,1]
    order_to_sort = np.argsort(ims.astype('int32'))
    return ims[order_to_sort], preds[order_to_sort]

def save_model_bone(model, fname):
    json_string = model.to_json()
    with open(fname, 'w') as f:
        json.dump(json_string, f)

def connect_models(model_1, model_2):
    return Model(inputs=model_1.input, outputs=model_2(model_1.output))

def schedule(epoch, base_lr, decay=0.5, staircase=True, steps=10):
    global which_step
    which_step = 0
    if staircase:
        if ((epoch+1)%steps == 0):
            print "decay learning rate by {}.".format(decay)
            which_step += 1
        return base_lr * decay**which_step
    else:
        return base_lr * decay**epoch

class getBestLossEpoch(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        self.best_loss = 1e2
        self.best_epoch = -1
    def on_epoch_end(self, epoch, logs={}):
        loss_the_epoch = logs.get('loss')
        if loss_the_epoch < self.best_loss:
                self.best_loss = loss_the_epoch
                self.best_epoch = epoch
                print "The best loss evaluation so far."

class customValidation(Callback):
    def __init__(self, validation_data, pre_net, interval=1):
        super(Callback, self).__init__()
        self.best_loss = 1e2
        self.best_epoch = -1
        self.pre_net = pre_net
        self.interval = interval
        if validation_data:
            self.x_val, self.y_val = validation_data
        else:
            raise ValueError('validation_data must be a tuple (x_val, y_val)')
    def on_epoch_end(self, epoch, logs={}):
        net = connect_models(self.pre_net, self.model)
        if epoch % self.interval == 0:
            y_pred = net.predict(self.x_val, verbose=0)
            val_loss = log_loss(self.y_val, y_pred)
            val_acc = accuracy_score(self.y_val, y_pred)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
            print "\n epoch: {:d} - val_loss: {:.6f}".format(epoch+1, val_loss)
            print "\n epoch: {:d} - val_acc: {:.6f}".format(epoch+1, val_acc)

def plot_history(history, fname):
    # plot history of loss and acc
    accs = history.history['acc']
    val_accs = history.history['val_acc']
    losses = history.history['loss']
    val_losses = history.history['val_loss']
    epochs = range(len(accs))
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, accs)
    ax1.plot(epochs, val_accs)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'val'], loc='best')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, losses)
    ax2.plot(epochs, val_losses)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend(['train', 'val'], loc='best')
    fig.savefig(fname)

def plot_confusion_matrix(cm, classes, fname, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.style.use('ggplot')
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname)
