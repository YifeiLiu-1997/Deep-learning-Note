from thumb_classifier_with_TensorFlow import predict_my_thumb
import pickle

# get parameters
with open("./result/2021-06-25_res.pickle", "rb") as fp:
    parameters = pickle.load(fp)


for i in range(0, 3):
    predict_my_thumb('/lyf_norm_{}.jpg'.format(str(i)), parameters)
