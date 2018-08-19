# Train the FC layers from bottleneck features of pre-trained models
# hyper-parameters:
#     optimizers (sgd/adam), lr, schedule, epochs, early-stopping
# data src:
#     training data, augmented data, pseudo-labelling
from data_utils import *
from utils import *

output_path = "/output/"
train_path = "/train_files/"
test_path = "/test_files/test/"
aug_path = "/aug_files/"
ps_path = "/ps_files/"
model_path = "/models/"  # for pre-loading trained model
dim_ordering = K.image_dim_ordering()
input_shape = (224, 224, 3) if dim_ordering == "tf" else (3, 224, 224)
batch_size = 64
base_lr = 5e-5

def load_data(data_src='normal'):
    x_train = load_array(train_path + 'x_train.bc')
    y_train = load_array(train_path + 'y_train.bc')
    x_val = load_array(train_path + 'x_val.bc')
    y_val = load_array(train_path + 'y_val.bc')
    if data_src == 'normal':
        return x_train, y_train, x_val, y_val
    elif data_src == 'augmentation':
        x_aug_1 = load_array(aug_path + 'x_aug.bc')
        y_aug_1 = load_array(aug_path + 'y_aug.bc')
        return x_train, y_train, x_val, y_val, x_aug, y_aug
    elif data_src == 'pseudo':
        x_ps = load_array(ps_path + 'x_ps.bc')
        y_ps = load_array(ps_path + 'y_ps.bc')
        return x_train, y_train, x_val, y_val, x_ps, y_ps

def get_fc(input_shape, pre_model='vgg'):
    if pre_model == 'vgg':
        inp = Input(shape=input_shape, name='fc_input')
        flatten = Flatten(name='flatten')(inp)
        bn_fc = BatchNormalization(name='bn_fc1')(flatten)
        inp_dropout = Dropout(0.65)(bn_fc)
        fc = Dense(1024, kernel_initializer='he_normal', name='fc1')(inp_dropout)
        bn_pred = BatchNormalization(name='bn_pred')(fc)
        fc_out = Activation('relu')(bn_pred)
        fc_dropout = Dropout(0.5)(fc_out)
        prediction = Dense(2, activation='softmax', name='prediction')(fc_dropout)
        model = Model(inp, prediction, name='vgg_fc_model')
        for L in model.layers[:-8]: L.trainable = False
    elif pre_model == 'resnet':
        inp = Input(shape=input_shape, name='fc_input')
        x = Flatten()(inp)
        prediction = Dense(2, activation='softmax', name='prediction')(x)
        model = Model(inp, prediction, name='resnet_fc_model')
    elif pre_model == 'inception':
        inp = Input(shape=input_shape, name='fc_input')
        x = GlobalAveragePooling2D(name='avg_pool')(inp)
        prediction = Dense(2, activation='softmax', name='prediction')(x)
        model = Model(inp, prediction, name='inception_fc_model')
    return model

checkpoint = ModelCheckpoint(output_path+'weights_fc.h5',
                             verbose=1, save_best_only=True,
                             save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
schedule.func_defaults = (base_lr, 0.5, True, 10)
scheduler = LearningRateScheduler(schedule)
nb_epoch = 40

def run(pre_model='resnet', optimizer='sgd', data_src='normal', pre_load=False):
    # prepare data
    if data_src == 'normal':
        x_train, y_train, x_val, y_val = load_data(data_src=data_src)
        y_train, y_val = onehot(y_train, 2), onehot(y_val, 2)
    elif data_src == 'augmentation':
        x_train, y_train, x_val, y_val, x_aug, y_aug = load_data(data_src=data_src)
        y_train, y_val = onehot(y_train, 2), onehot(y_val, 2)
        x_train = np.concatenate([x_train, x_aug])
        y_train = np.concatenate([y_train, y_aug])
    elif data_src == 'pseudo':
        x_train, y_train, x_val, y_val, x_ps, y_ps = load_data(data_src=data_src)
        y_train, y_val, y_ps = onehot(y_train, 2), onehot(y_val, 2), onehot(y_ps, 2)
        x_train = np.concatenate([x_train, x_ps])
        y_train = np.concatenate([y_train, y_ps])

    # prepare fc and train
    fc = get_fc(x_train.shape[1:], pre_model=pre_model)
    if pre_load:
        fc.load_weights(model_path + 'weights_fc.h5')
    save_model_bone(fc, output_path+'fc.json')
    print "FC structure:", fc.summary()
    callbacks = [checkpoint, earlystopping, scheduler]
    if optimizer == 'sgd':
        optim = SGD(lr=base_lr, momentum=0.9, decay=0., nesterov=True)
    elif optimizer == 'adam':
        optim = Adam(lr=base_lr)
    fc.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
    history = fc.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size,
                     validation_data=(x_val, y_val), callbacks=callbacks)
    fc.load_weights(output_path + 'weights_fc.h5')

    # complete model for test
    if pre_model == 'vgg':
        pre_net = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif pre_model == 'resnet':
        pre_net = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif pre_model == 'inception':
        pre_net = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    model = connect_models(pre_net, fc)
    print "Whole model structure:", model.summary()
    ims, preds = predict(test_path, model)
    submit = pd.DataFrame({'id':ims, 'label':preds})
    submit.to_csv(output_path + 'catsdogs_fc.csv', index=False)

    # plot history of fc
    plot_history(history, output_path+'catsdogs_fc.png')

if __name__ == "__main__":
    run()
