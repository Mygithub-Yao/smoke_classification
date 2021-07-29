import tensorflow as tf
def vgg_11_Sequential():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))

    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Reshape(target_shape=(7 * 7 * 512,)))
    model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
    model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
    model.add(tf.keras.layers.Dense(units=3))

    return model