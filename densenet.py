from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,classes=1000, weights_path=None):
    eps = 1.1e-5

    # 압축 정도
    compression = 1.0 - reduction

    global concat_axis
    concat_axis = 3
    img_input = Input(shape=(224,224,3), name='data')

    nb_filter = 64

    # densenet 버전에 따른 수정가능
    nb_layers = [6,12,32,32]             # densenet 169

    # 초기 convolution
    x = ZeroPadding2D((3,3))(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2,2), bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        x = transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    x, nb_filter =  dense_block(x, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(classes)(x)
    x = Activation('softmax')(x)

    model = Model(img_input, x)

    return model


def conv_block(x, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    eps = 1.1e-5

    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis= concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(inter_channel, 1, 1, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(nb_filter, 3, 3, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition_block(x, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):

    eps = 1.1e-5

    x = BatchNormalization(epsilon=eps, axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2,2), strides=(2,2))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter