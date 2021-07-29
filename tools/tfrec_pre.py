import tensorflow as tf
import os
import numpy as np
import random

class tfrec_pre:
    def __init__(self,rootpath,train_val_percentage=None,train_val_abs=None,resize=None):
        self.dataroot = rootpath
        self.tfrecord_train = os.path.join(rootpath , 'train.tfrecords')
        self.tfrecord_val = os.path.join(rootpath , 'val.tfrecords')
        self.train_val_percentage = train_val_percentage
        self.train_val_abs = train_val_abs
        self.resize = resize
        
        self.classlist = []
        # 定义Feature结构，告诉解码器每个Feature的类型是什么
        self.feature_description = { 
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

        
    def _parse_example_jpeg(self,example_string):
        # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, self.feature_description)
        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'],channels=3)    # 解码JPEG图片
        return feature_dict['image'], feature_dict['label']
    
    def _parse_example_png(self,example_string):
        feature_dict = tf.io.parse_single_example(example_string, self.feature_description)
        feature_dict['image'] = tf.io.decode_png(feature_dict['image'],channels=3)    # 解码PNG图片
        return feature_dict['image'], feature_dict['label']
    
    def _resize_img(self,img,label):
        #图片归一
        image_resized = tf.image.resize(img, self.resize) / 255.0
        return image_resized,label
    
    def _Data_Augmentation(self,image,label):
         #随机水平翻转图像
        #image=tf.image.random_flip_left_right(img)
        #随机改变图像的亮度
        image=tf.image.random_brightness(image,0.1)
        #随机改变对比度
        image=tf.image.random_contrast(image,0.9,1.1)
        #随机改变饱和度
        image = tf.image.random_saturation(image,0.9,1.1)
        #随机裁剪
        #image = tf.random_crop(image,[120,120,3])
        #随机改变色调
        image = tf.image.random_hue(image,0.1)
        return image,label
    
    
    #只有train集
    def _tfrec_writer(self,train_filenames,train_labels):
        with tf.io.TFRecordWriter(self.tfrecord_train) as writer:
            for filename, label in zip(train_filenames, train_labels):
                image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
                feature = {                             # 建立 tf.train.Feature 字典
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
                writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
        
    #按绝对比例划分（train，val）                  
    def _tfrec_writer_abs(self,train_filenames,train_labels,absper=None):
        dataset=list(zip(train_filenames,train_labels))
        random.shuffle(dataset) 
        data_num = len(dataset)       
        train_num = int(data_num * absper)
        
        with tf.io.TFRecordWriter(self.tfrecord_train) as train:
            print('开始写入train集')
            for (filename,label) in dataset[:train_num-1]:
                
                image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
                feature = {                             # 建立 tf.train.Feature 字典
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
                train.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
            
        with tf.io.TFRecordWriter(self.tfrecord_val) as val:
            print('开始写入val集')
            for (filename,label) in dataset[train_num:]:
                
                image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
                feature = {                             # 建立 tf.train.Feature 字典
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
                val.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
                            
    #按概率划分（train，val）                        
    def _tfrec_writer_percentage(self,train_filenames,train_labels,percentage=None):
        with tf.io.TFRecordWriter(self.tfrecord_train) as train:
            with tf.io.TFRecordWriter(self.tfrecord_val) as val:
                #choices = np.random.choice([0, 1], size=1000, p=[percentage, 1-percentage])
                for (filename,label) in zip(train_filenames, train_labels):
                    image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
                    feature = {                             # 建立 tf.train.Feature 字典
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
                    choice=np.random.choice([0, 1],p=[percentage, 1-percentage])
                    if choice==0:
                        train.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
                    else:
                        val.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
        
    def generate(self):
        train_filenames = []
        train_labels = []
        
        for root,dirs,files in os.walk(self.dataroot):
            
            for dirname in dirs:
                #将目录名作为分类
                self.classlist.append(dirname)
                
            if os.path.split(root)[-1] in self.classlist:
                #获取目录名
                classname=os.path.split(root)[-1]
                new_filenames = [os.path.join(root,filename) for filename in files]
                train_filenames = train_filenames+new_filenames
                #找到目录名对应的下标作为这类别的标签
                train_labels = train_labels +[self.classlist.index(classname)] * len(new_filenames)

                
        if self.train_val_percentage == None and self.train_val_abs==None:
            self._tfrec_writer(train_filenames,train_labels)
        if self.train_val_percentage == None and self.train_val_abs!=None:
            self._tfrec_writer_abs(train_filenames,train_labels,self.train_val_abs)
        if self.train_val_percentage != None and self.train_val_abs==None:
            self._tfrec_writer_percentage(train_filenames,train_labels,self.train_val_percentage)
        if self.train_val_percentage != None and self.train_val_abs!=None:
            raise RuntimeError('不能同时使用参数train_val_abs和train_val_percentage')
        
    def load_tfrec_jpeg(self,filename):
        raw_dataset = tf.data.TFRecordDataset(os.path.join(self.dataroot,filename))    # 读取 TFRecord 文件
        dataset = raw_dataset.map(self._parse_example_jpeg,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._resize_img,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def load_tfrec_png(self,filename):
        raw_dataset = tf.data.TFRecordDataset(os.path.join(self.dataroot,filename))    # 读取 TFRecord 文件
        dataset = raw_dataset.map(self._parse_example_png)
        dataset = dataset.map(self._resize_img)
        return dataset

    
    def load_tfrec_augdata(self,filename):
        raw_dataset = tf.data.TFRecordDataset(os.path.join(self.dataroot,filename))    # 读取 TFRecord 文件
        dataset = raw_dataset.map(self._parse_example_png,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._resize_img,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._Data_Augmentation,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset