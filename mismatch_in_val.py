from data_utils import *
from utils import *

output_path = "/output/"
val_path = "/train_files/valid/"
fc_path = "/fc_model/"
CLASSES = ['CAT','DOG']

with open(fc_path+'fc.json') as f:
    fc_json = json.load(f)
fc = model_from_json(fc_json)
fc.load_weights(fc_path+'weights_fc.h5')
#pre_net = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
pre_net = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
model = connect_models(pre_net, fc)

val_files = class_files(val_path)
x_val, y_val = load_image_label(val_path, files=val_files)
pval = model.predict(x_val, batch_size=64, verbose=1)[:,1]
y_pval = np.array([1 if p>=0.5 else 0 for p in pval])

# plot the confusion matrix
cm = confusion_matrix(y_val, y_pval)
plot_confusion_matrix(cm, CLASSES, output_path+'cm.png')

# find those not matched
not_match_id = np.where(y_val != y_pval)[0]
not_match_prob = pval[not_match_id]
not_match = []
size_per_class = len(y_val) / 2
for idx in not_match_id:
    if idx < size_per_class:
        not_match.append(val_files['CAT'][idx])
    else:
        not_match.append(val_files['DOG'][idx-size_per_class])

with open(output_path+'mismatch_val.pkl', 'wb') as f:
    pickle.dump(not_match, f)
with open(output_path+'mismatch_prob_val.pkl', 'wb') as f:
    pickle.dump(not_match_prob, f)

# find those matched
match_id = np.where(y_val == y_pval)[0]
match_prob = pval[match_id]

# find those matched but uncertain
uncertain_id = np.where((match_prob<0.7)&(match_prob>0.3))[0]
uncertain_prob = match_prob[uncertain_id]
uncertain_id = match_id[uncertain_id]
uncertain = []
for idx in uncertain_id:
    if idx < size_per_class:
        uncertain.append(val_files['CAT'][idx])
    else:
        uncertain.append(val_files['DOG'][idx-size_per_class])
with open(output_path+'uncertain_val.pkl', 'wb') as f:
    pickle.dump(uncertain, f)
with open(output_path+'uncertain_prob_val.pkl', 'wb') as f:
    pickle.dump(uncertain_prob, f)

