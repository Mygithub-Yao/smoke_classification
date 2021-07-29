# coding=utf-8
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import models.vgg_16_optim_Sequential as vgg
from tools import tfrec_pre as tfrec_loader
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # data directories
    parser.add_argument('--train', type=str, default=r'./dataset/train')
    parser.add_argument('--test', type=str, default=r'./dataset/train')
   
    #模型输出路径
    parser.add_argument('--model_dir', type=str, default=r'./Output')
    
    return parser.parse_known_args()
    
    
def one_hot_map(img,label):
    one_hot_label=tf.one_hot(label,depth=3)
    return img,one_hot_label
    
    
def get_train_data(train_dir):  
    dataset = tfrec_loader.tfrec_pre(train_dir,resize=[128,128]).load_tfrec_augdata('train_AUG.tfrecords').map(one_hot_map)
    dataset = dataset.shuffle(buffer_size=1000)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    
    
def get_test_data(test_dir):
    dataset = tfrec_loader.tfrec_pre(test_dir,resize=[128,128]).load_tfrec_png('val_AUG.tfrecords').map(one_hot_map)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
                    
#y_pred已经经过softmax函数，并且y_true进行one-hot编码的
def focal_loss(y_true, y_pred,gamma=2.0, alpha=0.25):
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0)
    y_true = tf.cast(y_true, tf.float32)
    loss = -y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
    loss = tf.reduce_sum(loss, axis=1)
    return loss

def train(args):
    train_set=get_train_data(args.train).batch(args.batch_size)
    val_set=get_test_data(args.test).batch(args.batch_size)
    save_path = os.path.join(args.model_dir , 'model_vgg16_focal.h5')
    try:
        model = tf.keras.models.load_model(save_path)
        #model = tf.keras.models.load_model(save_path,custom_objects={'focal_loss': focal_loss})
    except Exception as e:
        print('#######Exception', e)
        model = vgg.vgg_16_optim_Sequential()
        
    #model = vgg.keras_sm_vgg(3)   
    #model = vgg.vgg_11_Sequential()
    #model = vgg.vgg_16_optim_Sequential()
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
               initial_learning_rate=args.learning_rate, decay_steps=500, decay_rate=0.96)
    
    
    # 设置指数衰减的学习率。
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=exponential_decay),
        loss=focal_loss,
        metrics=['accuracy']
    )
    model.fit(train_set, epochs=args.epochs,validation_data=val_set)
    #保存为h5文件的模型
    new_save_path = os.path.join(args.model_dir , 'model_vgg16_focal.h5')
    #保存为标准的pd格式模型
    #new_save_path = os.path.join(args.model_dir , 'model_vgg16_focal')
    model.save(new_save_path)
    
    
    #------------------------学习率衰减----------------------------------
    #exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    #           initial_learning_rate=args.learning_rate, decay_steps=1000, decay_rate=0.96)
    
    
    #-------------------其他可选的搭配-------------------------------------
    #tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #tf.keras.optimizers.Adam(learning_rate=exponential_decay)
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #loss=tf.keras.losses.sparse_categorical_crossentropy,
    #metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    #metrics=['accuracy']
    
    
    
if __name__ == '__main__':
    args,_ = parse_args()
    train(args)