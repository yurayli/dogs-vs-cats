# Train the FC layers from bottleneck features of pre-trained models
# hyper-parameters:
#     optimizers (sgd/adam), lr, schedule, epochs, early-stopping
# data src:
#     training data, augmented data, pseudo-labelling
from data_utils import *
from utils import *

output_path = "/output/"
val_path = "/train_files/valid/"
test_path = "/train_files/test/"
nb_model = 4
model_paths = ["/model_{}/".format(i+1) for i in range(nb_model)]
input_shape = (224, 224, 3)
batch_size = 64

def load_data():
    val_files = class_files(val_path)
    x_val, y_val = load_image_label(val_path, files=val_files)
    return x_val, y_val

def load_model(path, pre_model):
    if pre_model == 'vgg':
        pre_net = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif pre_model == 'resnet':
        pre_net = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    with open(path+'fc.json') as f:
        fc_json = json.load(f)
    fc = model_from_json(fc_json)
    fc.load_weights(path+'weights_fc.h5')
    return connect_models(pre_net, fc)

def run():
    # prepare data
    x_val, y_val = load_data()
    print x_val.shape, y_val.shape

    # calculate preds of validation
    preds_val, preds_test = [], []
    for i in range(len(model_paths)):
        if i < 2:
            m = load_model(model_paths[i], 'vgg')
        else:
            m = load_model(model_paths[i], 'resnet')
        preds_val.append(m.predict(x_val, batch_size=batch_size, verbose=1)[:,1])
        preds_test.append(predict(test_path, m)[1])
    preds_val = np.stack(preds_val).T
    preds_test = np.stack(preds_test).T
    print preds_val.shape, preds_test.shape
    test_size = len(preds_test)

    # calculate the optimal weights for ensembling
    '''
    losses = np.array([log_loss(preds_val[:,i], y_val) for i in range(len(model_paths))])
    losses /= np.sum(losses)
    pred = np.sum([preds_test[:,i]*losses[i] for i in range(len(model_paths))], 0)
    submit = pd.DataFrame({'id':np.arange(test_size)+1, 'label':pred})
    submit.to_csv(output_path + 'ensemble_from_loss.csv', index=False)

    accs = np.array([accuracy_score([1 if p>=0.5 else 0 for p in preds_val[:,i]], y_val) \
                     for i in range(len(model_paths))])
    accs /= np.sum(accs)
    pred = np.sum([preds_test[:,i]*accs[i] for i in range(len(model_paths))], 0)
    submit = pd.DataFrame({'id':np.arange(test_size)+1, 'label':pred})
    submit.to_csv(output_path + 'ensemble_from_acc.csv', index=False)

    inputs = K.placeholder(shape=(len(y_val), nb_model))
    w = K.zeros(shape=(nb_model,))
    out = K.dot(inputs, w)
    loss = K.mean((y_val*K.log(out) + (1-y_val)*K.log(1-out)), 0)
    grads = K.gradients(loss, inputs)[0]
    '''
    lr = LogisticRegression(fit_intercept=False)
    lr.fit(preds_val, y_val)
    print lr.coef_
    pred = lr.predict_proba(preds_test)[:,1]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    submit = pd.DataFrame({'id':np.arange(test_size)+1, 'label':pred})
    submit.to_csv(output_path + 'ensemble_from_fit.csv', index=False)

if __name__ == "__main__":
    run()
