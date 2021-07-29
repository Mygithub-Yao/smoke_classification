import tensorflow as tf
class VGG_16_class(tf.keras.Model):
    def __init__(self):
        '''
            图片尽量用大点，因为卷积层和池化层比较多
        '''
        super().__init__()
        # --------------------conv3-64，conv3-64，maxpool(2*2)--------------------
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        # --------------------conv3-128，conv3-128，maxpool(2*2)--------------------
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        # --------------------conv3-256，conv3-256，conv1-256,maxpool(2*2)--------------------
        self.conv5 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv6 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv7 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[1, 1],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        # --------------------conv3-512，conv3-512，conv1-512,maxpool(2*2)--------------------
        self.conv8 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv9 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv10 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[1, 1],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        # --------------------conv3-512，conv3-512，conv1-512,maxpool(2*2)--------------------
        self.conv11 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv12 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu
        )
        self.conv13 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[1, 1],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        # -------------------------展开--------------------------------
        # self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 512,))
        self.flatten = tf.keras.layers.Flatten()

        # -------------------------全连接-------------------------------
        self.dense1 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=1000,activation = tf.nn.softmax)

        # self.dense4 = tf.keras.layers.Dense(units=10)

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 224, 224, 64]
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)  # [batch_size, 112, 112, 128]
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)  # [batch_size, 56, 56, 256]
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)  # [batch_size, 28, 28, 512]
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)

        x = self.conv11(x)  # [batch_size, 14, 14, 512]
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)

        x = self.flatten(x)  # [batch_size, 7 * 7 * 512]

        x = self.dense1(x)  # [batch_size, 4096]
        x = self.dense2(x)  # [batch_size, 4096]
        output = self.dense3(x)  # [batch_size, 1000]
        # x = self.dense4(x)
        # output = tf.nn.softmax(x)
        return output