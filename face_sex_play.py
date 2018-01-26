import numpy as np
from cnn_mate import SimpleConvNet
import cv2
from layers import *


def face_ageer(x):
    image = cv2.imread(x)
    input_image = []
    input_image.append(cv2.resize(image, (32, 32)))
    #image = image.astype(np.float32)
    #image /= 255.0
    arr = [sky.T for sky in input_image]
    input_image = np.array(arr)

    #print(arr.shape)

    network = SimpleConvNet(input_dim=(3, 32, 32),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=1000, output_size=10, weight_init_std=0.01)

    network.load_params("params_epoch10_age.pkl")

    answer = softmax(network.predict(input_image))

    age_number = 0
    for i in range(10):
        if(answer[0, i]>answer[0, i-1]):
            age_number = i

    return age_number
