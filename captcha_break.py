import base64
import json
import sys
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Lambda
from tensorflow.keras.layers import GRU, LSTM, Bidirectional
from tensorflow.keras.layers import Add, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as K
import tensorflow as tf

print("Tensorflow version: ", tf.__version__)

import string
import cv2 as cv

def captcha_predict(file_name):

    letters = set()
    for ch in string.ascii_uppercase:
        letters.add(ch)
    letters = sorted(letters)

    batch_size = 32

    img_width = 132
    img_height = 48

    downsample_factor = 4

    max_text_len = 4

    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_model(training=True):

        input_img = Input(shape=(img_width, img_height, 1), name='input_data', dtype='float32')
        labels = Input(name='input_label', shape=[max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # print(input_img.shape)
        # feaure extraction
        x0 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1_1')(input_img)
        x = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1_2')(x0)
        x = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1_3')(x)
        # x = x+x0
        x = Concatenate(axis=-1)([x0,x])
        x = Conv2D(32,(1,1), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1_4')(x)
        x = Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1_5')(x)
        X = BatchNormalization()(x, training=training)
        x = MaxPooling2D((2,2), name='pool1')(x)
        x1 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2_1')(x)
        x = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2_2')(x)
        x = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2_3')(x)
        x = Concatenate(axis=-1)([x1,x])
        x = Conv2D(64, (1,1), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2_4')(x)
        x = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2_5')(x)
        x = BatchNormalization()(x, training=training)
        x = MaxPooling2D((2,2), name='pool2')(x)
        

        new_shape = ((img_width // 4), (img_height // 4)*64)
        x = Reshape(target_shape=new_shape, name='reshape')(x)
        x = Dense(64, activation='relu', name='dense1')(x)
        
        # RNNs
        x = Bidirectional(LSTM(128, return_sequences=True,  name='lstm_1'), name='bi_1')(x)
        x = Bidirectional(LSTM(128, return_sequences=True,  name='lstm_2'), name='bi_2')(x)
        
        # final part
        x = Dense(len(letters)+1, activation='softmax', name='dense2', kernel_initializer='he_normal')(x)
        
        # Get the CTC loss
        output = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([x, labels, input_length, label_length])
        
        #the final model
        model = Model([input_img, labels, input_length, label_length], output, name='ocr_model_v1')
        
        # optimizer
        sgd = SGD(learning_rate=0.01, decay=1e-5, momentum=0.9, nesterov=True, clipnorm=5)
        
        model.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        return model

    model = build_model()
    model.load_weights("captcha_cracker.h5")

    def decode_batch_predictions(pred):
        pred = pred[:, 2:]
        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        results = K.get_value(K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0])

        texts = []
        for res in results:
            outstr = ""
            for c in res:
                if c < len(letters):
                    outstr += letters[c]
            texts.append(outstr)

        return texts

    def imread(path_to_img, img_width=132, img_height=48):
        im = cv.imread(str(path_to_img), cv.IMREAD_GRAYSCALE)
        im = cv.resize(im, (img_width, img_height))
        im = (im/255.).astype(np.float32)
        im = np.transpose(im,(1,0))
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=-1)
        return im

    output_func = K.function([model.get_layer(name='input_data').input],
                        [model.get_layer(name='dense2').output])

    in_data = imread(file_name)
    preds = output_func(in_data)[0]

    pred_texts = decode_batch_predictions(preds)[0]

    return pred_texts[:4]
    
# def base64_api(uname, pwd, img, typeid):
#     base64_data = base64.b64encode(img)
#     b64 = base64_data.decode()
#     data = {"username": uname, "password": pwd, "typeid": typeid, "image": b64}
#     result = json.loads(requests.post("http://api.ttshitu.com/predict", json=data).text)
#     return result


def reportError(id):
    data = {"id": id}
    result = json.loads(requests.post("http://api.kuaishibie.cn/reporterror.json", json=data).text)
    if result["success"]:
        return "报错成功"
    else:
        return result["message"]


from sys import exit as sys_exit


def getCaptchaData(zlapp):
    url = "https://zlapp.fudan.edu.cn/backend/default/code"
    headers = {
        "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "accept-encoding": "gzip",
        "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
        "dnt": "1",
        "referer": "https://zlapp.fudan.edu.cn/site/ncov/fudanDaily",
        "sec-ch-ua": '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
        "sec-ch-ua-mobile": "?0",
        "sec-fetch-dest": "image",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-origin",
        "User-Agent": zlapp.UA,
    }
    res = zlapp.session.get(url, headers=headers)
    return res.content


class DailyFDCaptcha:
    zlapp = None
    uname = ""
    pwd = ""
    typeid = 2  # 纯英文
    info = lambda x: x
    id = 0

    def __init__(self, uname, pwd, zlapp, info_callback):
        self.zlapp = zlapp
        self.uname = uname
        self.pwd = pwd
        self.info = info_callback

    def __call__(self):
        img = getCaptchaData(self.zlapp)

        file_name = 'captcha.jpg'
        with open(file_name, "wb") as f:
            f.write(img)

        captcha_text = captcha_predict(file_name)
        return captcha_text
        # result = base64_api(self.uname, self.pwd, img, self.typeid)
        # print(result)
        # if result["success"]:
        #     self.id = result["data"]["id"]
        #     return result["data"]["result"]
        # else:
        #     self.info(result["message"])

    def reportError(self):
        if self.id != 0:
            self.info(reportError(self.id))


# if __name__ == "__main__":

#     def base64_api(uname, pwd, img, typeid):
#         return {"success": False, "code": "-1", "message": "用户名或密码错误", "data": ""}

#     print(base64_api(0, 0, 0, 0))
#     test = DailyFDCaptcha(0, 0, 0, print)
#     test(0)

#     def base64_api(uname, pwd, img, typeid):
#         return {
#             "success": True,
#             "code": "0",
#             "message": "success",
#             "data": {"result": "hhum", "id": "00504808e68a41ad83ab5c1e6367ae6b"},
#         }

#     print(test(0))

#     def reportError(id):
#         return id

#     test.reportError()
