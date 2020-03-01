import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
from keras.models import Model


def showmiddle():
    tmpfile = os.path.join("..", "data", "maskpaper", "val", "image00291.jpg")
    image = skimage.io.imread(tmpfile)
    ins = Frame()
    ins.load_paper()
    # print(ins.model_paper.keras_model)
    # print('\n'.join(['%s:%s' % item for item in ins.model_paper.keras_model.__dict__.items()]))
    # for i1 in range(1000):
    #     print(i1)
    #     print(ins.model_paper.keras_model.get_layer(index=i1))
    #     print(ins.model_paper.keras_model.get_layer(index=i1).output)
    conv1_layer = Model(inputs=ins.model_paper.keras_model.get_layer(index=0).output,
                        outputs=ins.model_paper.keras_model.get_layer(index=1).output)
    conv1_output = conv1_layer.predict(np.expand_dims(image, 0))
    print(conv1_output)
    batchsize = 5
    for i in range(batchsize):
        print(i)
        show_img = conv1_output[i, :, :, :]
        plt.imshow(image, interpolation='nearest')
        plt.show()
        plt.imshow(show_img, interpolation='nearest')
        plt.show()


def origin_showmiddle():
    input_data = Input(shape=(28, 28, 1))
    x = Conv2D(64, kernel_size=3, padding='same', activation='relu', name='conv1')(input_data)
    x = MaxPooling2D(pool_size=2, strides=2, name='maxpool1')(x)
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='maxpool2')(x)
    x = Dropout(0.25)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(10, activation='softmax', name='fc2')(x)
    model = Model(inputs=input_data, outputs=x)

    model.load_weights('final_model_mnist_2019_1_28.h5')

    raw_img = cv2.imread('test.png')
    test_img = load_img('test.png', color_mode='grayscale', target_size=(28, 28))
    test_img = np.array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = np.expand_dims(test_img, axis=3)

    conv1_layer = Model(inputs=input_data, outputs=model.get_layer(index=1).output)

    conv1_output = conv1_layer.predict(test_img)

    for i in range(64):
        show_img = conv1_output[:, :, :, i]
        print(show_img.shape)
        show_img.shape = [28, 28]
        cv2.imshow('img', show_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    showmiddle()
