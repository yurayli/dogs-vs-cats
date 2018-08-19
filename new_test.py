from data_utils import *
from utils import *

fc_path = "/fc_model/"

with open(fc_path+'fc.json') as f:
    fc_json = json.load(f)
fc = model_from_json(fc_json)
fc.load_weights(fc_path+'weights_fc.h5')
#pre_net = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
pre_net = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
model = connect_models(pre_net, fc)

test_files = class_files('./images/')
x, y = load_image_label('./images/', files=test_files)
prob = model.predict(x, verbose=1)[:,1]
print test_files['CAT'], test_files['DOG']
print prob


