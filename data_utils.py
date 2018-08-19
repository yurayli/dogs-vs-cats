import os
import json
import itertools
import cPickle as pickle
import bcolz
import numpy as np
import pandas as pd
from random import shuffle

from scipy.misc import imread
from scipy.misc import imresize
import scipy.ndimage as ndi

import keras.backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

CLASSES = ['CAT','DOG']
dim_ordering = K.image_dim_ordering()
input_shape = (224, 224, 3) if dim_ordering == "tf" else (3, 224, 224)


# util function for image augmentation

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=2,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

class Generator(object):
    # Set batch_size at least 16
    def __init__(self, path_prefix, files, batch_size,
                 train_steps, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.05,
                 rotation_range=10,
                 do_crop=True,
                 crop_area_range=[0.8, 1.0],
                 aspect_ratio_range=[3./4., 4./3.],
                 seed=None):
        self.path_prefix = path_prefix
        self.trn_files = files
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_range = rotation_range
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.seed = seed

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        # a = 2*rn*var + (1-var)  in 0.5-1.5
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += (1 - self.saturation_var)
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        # a = 2*rn*var + (1-var)  in 0.5-1.5
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += (1 - self.saturation_var)
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        # a = 2*rn*var + (1-var)  in 0.5-1.5
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += (1 - self.contrast_var)
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, im):
        cov = np.cov(im.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        im += noise
        return np.clip(im, 0, 255)

    def horizontal_flip(self, im):
        if np.random.random() < self.hflip_prob:
            im = im[:, ::-1]
        return im

    def vertical_flip(self, im):
        if np.random.random() < self.vflip_prob:
            im = im[::-1]
        return im

    def rotation(self, im):
        theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
        h,w = im.shape[0], im.shape[1]
        transform_matrix = transform_matrix_offset_center(matrix, h, w)
        im = apply_transform(im, transform_matrix)
        return im

    def random_sized_crop(self, im):
        im_w = im.shape[1]
        im_h = im.shape[0]
        im_area = im_w * im_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * im_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, im_w)
        w_rel = w / im_w
        w = int(w)
        h = min(h, im_h)
        h_rel = h / im_h
        h = int(h)
        x = np.random.random() * (im_w - w)
        x_rel = x / im_w
        x = int(x)
        y = np.random.random() * (im_h - h)
        y_rel = y / im_h
        y = int(y)
        im = im[y:y+h, x:x+w]
        return im

    def generate(self, train=True):
        ## Load data from each class directory with corresponding class weights
        ## Each batch is retrieved by corresponding class_im_idx (updated for each generation)
        data_path = self.path_prefix
        gen_data = self.trn_files
        while True:
            inputs = []
            targets = []
            # shuffle the training files
            gen_data = {c:np.random.permutation(gen_data[c]) for c in CLASSES}
            class_im_idx = {'CAT':0,'DOG':0}
            for s in range(self.train_steps):
                for c in CLASSES:
                    nb_samples_to_load = self.batch_size//2
                    for k in range(nb_samples_to_load):
                        # load image and its target
                        im = load_img(data_path, gen_data[c], class_im_idx[c])
                        class_im_idx[c] = (class_im_idx[c] + 1) % len(gen_data[c])
                        #tar = 0 if c=='CAT' else 1
                        tar = np.zeros(len(CLASSES))
                        tar[np.array(CLASSES)==c] += 1
                        # augmentation on im
                        if self.seed:
                            np.random.seed(self.seed)
                        if train and self.do_crop:
                            im = self.random_sized_crop(im)
                        im = imresize(im, self.image_size).astype('float32')
                        if train:
                            shuffle(self.color_jitter)
                            for jitter in self.color_jitter:
                                im = jitter(im)
                            if self.lighting_std:
                                im = self.lighting(im)
                            if self.hflip_prob > 0:
                                im = self.horizontal_flip(im)
                            if self.vflip_prob > 0:
                                im = self.vertical_flip(im)
                            if self.rotation_range:
                                im = self.rotation(im)
                        inputs.append(im)
                        targets.append(tar)
                        if len(targets) == self.batch_size:
                            tmp_inp = np.array(inputs)
                            tmp_targets = np.array(targets)
                            inputs = []
                            targets = []
                            yield preprocess_input(tmp_inp), tmp_targets


def load_img(im_path, im_files, im_index):
    return imread( os.path.join(im_path, im_files[im_index]) ).astype('float32')

def data_size(fpath):
    return len(os.listdir(fpath))

def load_image_label(path, files=None, label=True):
    # return image arrays with/without corresponding labels
    inp_shape = input_shape[:-1] if dim_ordering == "tf" else input_shape[1:]
    if label:
        x = np.array([image.img_to_array(image.load_img(os.path.join(path,im), \
                     target_size=inp_shape)) for c in CLASSES for im in files[c]])
        y = np.array([1 if c=='DOG' else 0 for c in CLASSES for im in files[c]])
        return preprocess_input(x), y
    else:
        ims = os.listdir(path)
        x = np.array([image.img_to_array(image.load_img(os.path.join(path,im), \
                     target_size=inp_shape)) for im in ims])
        return ims, preprocess_input(x)

def onehot(y, num_classes):
    y = y.reshape(-1,1)
    return (np.arange(num_classes)==y).astype(np.int8)

def train_valid_split(features=None, labels=None, fpath=None, sample_split_ratio=0.1,
                      split_from='var', seed=30):
    if split_from=='dir':
        # split directory 'train' to 'train' and 'valid' (each with subdirs of each class)
        train_path = os.path.join(fpath, 'train')
        val_path = os.path.join(fpath, 'valid')
        if not os.path.exists(val_path): os.mkdir(val_path)
        for c in CLASSES:
            class_dir = os.path.join(val_path, c)
            if not os.path.exists(class_dir): os.mkdir(class_dir)
        size_per_class = [len(os.listdir(os.path.join(train_path, c))) for c in CLASSES]
        np.random.seed(seed)
        sample_idx_class = [np.random.choice(range(s), int(sample_split_ratio*s), replace=False) for s in size_per_class]
        for i in range(len(CLASSES)):
            ims_to_sample = np.array(os.listdir(os.path.join(train_path, CLASSES[i])))[sample_idx_class[i]]
            for im in ims_to_sample:
                os.rename(os.path.join(train_path, CLASSES[i], im), os.path.join(val_path, CLASSES[i], im))
    elif split_from=='files':
        # return train/val filenames
        im_files = class_files(fpath)
        trn_files, val_files = {}, {}
        for c in CLASSES:
            np.random.seed(seed)
            mask = np.random.rand(len(im_files[c])) < sample_split_ratio
            trn_files[c] = np.array(im_files[c])[~mask]
            val_files[c] = np.array(im_files[c])[mask]
        return trn_files, val_files
    elif split_from=='var':
        np.random.seed(seed)
        mask = np.random.rand(len(labels)) < sample_split_ratio
        x_train, y_train = features[~mask], labels[~mask]
        x_val, y_val = features[mask], labels[mask]
        return x_train, y_train, x_val, y_val

def class_files(fpath):
    im_files = {}
    for c in CLASSES:
        im_files[c] = [f for f in os.listdir(fpath) if f.startswith(c.lower())]
    return im_files

def save_array(fname, array):
    c = bcolz.carray(array, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=False, batch_size=64,
                class_mode='categorical', target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
                    class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def get_augment_data(generator, augment_times):
    x_aug, y_aug = [], []
    for i in range(augment_times):
        x_batch, y_batch = generator.next()
        x_aug.append(x_batch)
        y_aug.append(y_batch)
    return preprocess_input(np.concatenate(x_aug)), np.concatenate(y_aug)

