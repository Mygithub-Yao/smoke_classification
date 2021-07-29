import tensorflow as tf

def vgg_16_function():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(
        filters=64,  # 卷积层神经元（卷积核）数目
        kernel_size=[3, 3],  # 感受野大小
        padding='same',  # padding策略（vaild 或 same）
        activation=tf.nn.relu  # 激活函数
    )(inputs)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)

    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)

    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)

    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(units=1000,activation = tf.nn.softmax)(x)

    model = tf.keras.model(inputs=inputs, outputs=outputs)
    return model