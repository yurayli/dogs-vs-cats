from data_utils import *
from utils import *

output_path = "/output/"
train_path = "/train_files/train/"
val_path = "/train_files/valid/"
test_path = "/train_files/test/"
model_path = "/models/"  # for pre-loading trained model
dim_ordering = K.image_dim_ordering()
input_shape = (224, 224, 3) if dim_ordering == "tf" else (3, 224, 224)
batch_size = 64

#train_size = data_size(train_path+'CAT') + data_size(train_path+'DOG')
train_size = data_size(train_path)
train_steps = int(train_size) // batch_size
augment_times = train_steps * 2

def load_data(data_src='normal'):
    trn_files, val_files = class_files(train_path), class_files(val_path)
    if data_src=='augmentation':
        return trn_files, val_path
    x_train, y_train = load_image_label(train_path, files=trn_files)
    x_val, y_val = load_image_label(val_path, files=val_files)
    return x_train, y_train, x_val, y_val

def load_pre_net(pre_model, intermediate=False):
    # warning: intermediate output could be large in memory.
    if pre_model == 'vgg':
        blocks = [[4,5], [7,8,9], [11,12,13], [15,16,17]]
        model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        model.load_weights(model_path + "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        if intermediate:
            model = Sequential(model.layers[:blocks[-1][0]])
        return model
    elif pre_model == 'resnet':
        stages = [[12,10,10], [12,10,10,10], [12,10,10,10,10,10], [12,10,10]]
        model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        if intermediate:
            last_name = model.layers[-(sum(stages[-1])+2)].name
            model = Model(model.input, model.get_layer(last_name).output)
        return model
    elif pre_model == 'inception':
        return InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

def pseudo_labeling(label_file, confidence=0.5001):
    test_label = pd.read_csv(label_file)
    pseudo = test_label[ (test_label['label']>=confidence) | (test_label['label']<=(1-confidence)) ]
    pseudo['id'] = pseudo['id'].apply(lambda id: str(id)+'.jpg')
    pseudo['label'] = pseudo['label'].apply(lambda y: 1 if y>=0.5 else 0)
    ps_files = {}
    for i in range(2): ps_files[CLASSES[i]] = pseudo['id'][pseudo['label']==i].values
    x_ps, y_ps = load_image_label(test_path, files=ps_files)
    return x_ps, y_ps


def run(pre_model='vgg', intermediate=False, data_src='pseudo'):
    pre_net = load_pre_net(pre_model=pre_model, intermediate=intermediate)
    pre_net.summary()
    if data_src == 'normal':
        x_train, y_train, x_val, y_val = load_data()
        x_train = pre_net.predict(x_train, batch_size=batch_size, verbose=1)
        x_val = pre_net.predict(x_val, batch_size=batch_size, verbose=1)
        save_array(output_path + 'x_train.bc', x_train)
        save_array(output_path + 'y_train.bc', y_train)
        save_array(output_path + 'x_val.bc', x_val)
        save_array(output_path + 'y_val.bc', y_val)
        print x_train.shape, y_train.shape, x_val.shape, y_val.shape
    elif data_src == 'augmentation':
        mode = 'keras'  # 'keras' or 'custom'
        if mode == 'custom':
            trn_files, _ = load_data(data_src)
            gen = Generator(train_path, trn_files, batch_size, train_steps, input_shape[:-1])
            it = gen.generate()
            x_aug, y_aug = [], []
            for i in xrange(augment_times):
                ims, tars = it.next()
                x_aug.append(pre_net.predict(ims))
                y_aug.append(tars)
            x_aug, y_aug = np.concatenate(x_aug), np.concatenate(y_aug)
        else:
            data_gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                height_shift_range=0.1, shear_range=0.15, zoom_range=0.1,
                channel_shift_range=10., horizontal_flip=True, fill_mode='constant')
            generator = get_batches(train_path, data_gen, batch_size=batch_size)
            x_aug, y_aug = get_augment_data(generator, augment_times)
            x_aug = pre_net.predict(x_aug, batch_size=batch_size)
        save_array(output_path + 'x_aug.bc', x_aug)
        save_array(output_path + 'y_aug.bc', y_aug)
        print x_aug.shape, y_aug.shape
    elif data_src == 'pseudo':
        csv_file = './{}_fc.csv'.format(pre_model)
        x_ps, y_ps = pseudo_labeling(csv_file)
        x_ps = pre_net.predict(x_ps, batch_size=batch_size, verbose=1)
        save_array(output_path + 'x_ps.bc', x_ps)
        save_array(output_path + 'y_ps.bc', y_ps)
        print x_ps.shape, y_ps.shape

if __name__ == "__main__":
    run()
