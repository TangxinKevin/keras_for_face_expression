from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Activation
from keras.layers import SeparableConv2D, Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop


def Mini_Xception(input_shape, num_classes, l2_regularization=0.0001):

    regularization = l2(l2_regularization)
    input_image = Input(shape=input_shape)

    # block 1
    x = Conv2D(8, (3, 3), padding='same', use_bias=False,
               kernel_regularizer=regularization, name='B1_conv1')(input_image)
    x = BatchNormalization(name='B1_conv1_bn')(x)
    x = Activation('relu', name='B1_conv1_relu')(x)
    x = Conv2D(8, (3, 3), padding='same', use_bias=False,
               kernel_regularizer=regularization, name='B1_conv2')(x)
    x = BatchNormalization(name='B1_conv2_bn')(x)
    x = Activation('relu', name='B1_conv2_relu')(x)

    # block 2
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='B2_conv1')(x)
    residual = BatchNormalization(name='B2_conv1_bn')(residual)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False,
                        name='B2_sepconv1')(x)
    x = BatchNormalization(name='B2_sepconv1_bn')(x)
    x = Activation('relu', name='B2_sepconv1_relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False,
                        name='B2_sepconv2')(x)
    x = BatchNormalization(name='B2_sepconv2_bn')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same',
                  name='B2_maxpool')(x)
    x = Add([x, residual])


    # block 3
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='B3_conv1')(x)
    residual = BatchNormalization(name='B3_conv1_bn')(residual)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                        name='B3_sepconv1')(x)
    x = BatchNormalization(name='B3_sepconv1_bn')(x)
    x = Activation('relu', name='B3_sepconv1_relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False,
                        name='B3_sepconv2')(x)
    x = BatchNormalization(name='B3_sepconv2_bn')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same',
                  name='B3_maxpool')(x)
    x = Add([x, residual])


    # block 4
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='B4_conv1')(x)
    residual = BatchNormalization(name='B4_conv1_bn')(residual)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                        name='B4_sepconv1')(x)
    x = BatchNormalization(name='B4_sepconv1_bn')(x)
    x = Activation('relu', name='B4_sepconv1_relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False,
                        name='B4_sepconv2')(x)
    x = BatchNormalization(name='B4_sepconv2_bn')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same',
                  name='B4_maxpool')(x)
    x = Add([x, residual])

    # block 5
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False, name='B5_conv1')(x)
    residual = BatchNormalization(name='B5_conv1_bn')(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                        name='B5_sepconv1')(x)
    x = BatchNormalization(name='B5_sepconv1_bn')(x)
    x = Activation('relu', name='B5_sepconv1_relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False,
                        name='B5_sepconv2')(x)
    x = BatchNormalization(name='B5_sepconv2_bn')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same',
                  name='B5_maxpool')(x)
    x = Add([x, residual])

    x = Conv2D(512, (4, 4), strides=(1, 1), padding='same',
               name='B6_conv')(x)
    x = Flatten(name='B6_flatten')(x)
    x = Dense(num_classes, activation='softmax',
              name='B6_predict')(x)

    model = Model(inputs=input_image, outputs=x)
    print(model.summary())

    return model



