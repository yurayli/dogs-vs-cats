from data_utils import *
from utils import *

output_path = "/output/"
train_path = "/train_files/train/"
val_path = "/train_files/valid/"
test_path = "/train_files/test/"
fc_path = "/fc_model/"  # for pre-loading trained FCs
model_path = "/models/"  # for pre-loading trained model
dim_ordering = K.image_dim_ordering()
input_shape = (224, 224, 3) if dim_ordering == "tf" else (3, 224, 224)
batch_size = 64
base_lr = 1e-4
train_size = data_size(train_path)
train_steps = int(train_size) // batch_size

def get_vgg_model(pre_load=False):
    if pre_load:
        with open(model_path+'vgg_complete.json') as f:
            model_json = json.load(f)
        model = model_from_json(model_json)
        model.load_weights(model_path+'weights.h5')
        return model
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    blocks = [[4,5], [7,8,9], [11,12,13], [15,16,17]]
    # load FCs
    with open(fc_path+'fc.json') as f:
        fc_json = json.load(f)
    fc = model_from_json(fc_json)
    fc.load_weights(fc_path+'weights_fc.h5')
    # return model
    x = vgg.layers[ blocks[-1][0]-1 ].output
    for l in blocks[-1]: # add bn at block 5
        x = BatchNormalization(axis=-1, name='block5_bn_{}'.format(l))(x)
        x = vgg.layers[l](x)
    x = vgg.layers[ blocks[-1][-1]+1 ](x)
    model = Model(vgg.input, fc(x))
    for L in model.layers[ :blocks[-1][0] ]: L.trainable = False
    return model

def get_res_model(pre_load=False):
    if pre_load:
        with open(model_path+'res_complete.json') as f:
            model_json = json.load(f)
        model = model_from_json(model_json)
        model.load_weights(model_path+'weights.h5')
        return model
    res = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    stages = [[12,10,10], [12,10,10,10], [12,10,10,10,10,10], [12,10,10]] # num layers each stage
    # load FCs
    with open(fc_path+'fc.json') as f:
        fc_json = json.load(f)
    fc = model_from_json(fc_json)
    fc.load_weights(fc_path+'weights_fc.h5')
    # return model
    model = Model(res.input, fc(res.output))
    for L in model.layers[ :-(sum(stages[-1])+1) ]: L.trainable = False
    return model

earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
checkpoint = ModelCheckpoint(output_path+'weights.h5',
                             verbose=1, save_best_only=True,
                             save_weights_only=True)
scheduler = LearningRateScheduler(schedule)
nb_epoch = 20

def run(pre_model='resnet', optimizer='sgd', pre_load=False):
    # prepare data
    trn_files, val_files = class_files(train_path), class_files(val_path)
    x_val, y_val = load_image_label(val_path, files=val_files)
    y_val = onehot(y_val, 2)
    gen = Generator(train_path, trn_files, batch_size,
                    train_steps, (input_shape[0], input_shape[1]))

    # prepare model and train
    if pre_model == 'vgg':
        model = get_vgg_model(pre_load=pre_load)
        save_model_bone(model, output_path+'vgg_complete.json')
    elif pre_model == 'resnet':
        model = get_res_model(pre_load=pre_load)
        save_model_bone(model, output_path+'res_complete.json')
    print "Model structure:", model.summary()
    callbacks = [earlystopping, checkpoint]
    if optimizer == 'sgd':
        optim = SGD(lr=base_lr, momentum=0.9, decay=1e-4, nesterov=True)
    elif optimizer == 'adam':
        optim = Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(gen.generate(), gen.train_steps,
                                  nb_epoch, verbose=1,
                                  callbacks=callbacks,
                                  validation_data=(x_val, y_val),
                                  workers=1)
    model.load_weights(output_path + 'weights.h5')

    # test model
    ims, preds = predict(test_path, model)
    submit = pd.DataFrame({'id':ims, 'label':preds})
    submit.to_csv(output_path + 'catsdogs.csv', index=False)

    # plot history
    plot_history(history, output_path+'catsdogs.png')

if __name__ == "__main__":
    run()