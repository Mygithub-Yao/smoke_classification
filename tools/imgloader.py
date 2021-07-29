# coding=utf-8
# coding=utf-8
import tensorflow as tf
import os

class imgloader:
    def __init__(self,rootpath,batch_size=1):
        self.dataroot = rootpath
        self.classlist = []
        self.batch_size = batch_size

    def _decode_and_resize(self,filename):
        image_string = tf.io.read_file(filename)  # 读取原始文件
        # 解码JPEG图片(并且以三通道读入),
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [128, 128]) / 255.0
        return image_resized
    
    def _deimg_and_filemane(self,filename,filename2):
        image_string = tf.io.read_file(filename)  # 读取原始文件
        # 解码JPEG图片(并且以三通道读入),
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [128, 128]) / 255.0
        return image_resized,filename2
    
    def load(self):
        #名字张量
        filenames_Tensor = tf.constant([os.path.join(self.dataroot,filename) for filename in os.listdir(self.dataroot)])
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames_Tensor,))
        
        #map处理
        dataset = dataset.map(
            map_func=self._decode_and_resize, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)   
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_filename_and_img(self):
        #名字张量(不带路径)
        filenames_Tensor2 = tf.constant([filename for filename in os.listdir(self.dataroot)])
        #名字张量(带路径)
        filenames_Tensor1 = tf.constant([os.path.join(self.dataroot,filename) for filename in os.listdir(self.dataroot)])
        
        dataset = tf.data.Dataset.from_tensor_slices((filenames_Tensor1,filenames_Tensor2))
        dataset = dataset.map(
            map_func=self._deimg_and_filemane, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE) 
        dataset = dataset.batch(self.batch_size)
        return dataset
        