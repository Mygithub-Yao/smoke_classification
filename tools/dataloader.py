# coding=utf-8
import tensorflow as tf
import os

class dataloader:
    def __init__(self,rootpath,batch_size=32):
        self.dataroot = rootpath
        self.classlist = []
        self.batch_size = batch_size

    def _decode_and_resize(self,filename, label):
        image_string = tf.io.read_file(filename)  # 读取原始文件
        # 解码JPEG图片(并且以三通道读入),
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0

        # image_resized = tf.reshape(image_resized,shape=[256, 256,3])
        # image_resized = tf.reshape(image_resized,shape=[256, 256,1])
        return image_resized, label

    def load(self):
        filenameslist = []
        for root,dirs,files in os.walk(self.dataroot):
            
            for dirname in dirs:
                self.classlist.append(dirname)
                
            if os.path.split(root)[-1] in self.classlist:
                filenameslist.append(
                    tf.constant([os.path.join(root,filename) for filename in files]))
                
        #名字张量
        filenames_Tensor = tf.concat(filenameslist, axis=-1)
        
        #标签张量
        labels_Tensor = tf.constant(0,shape=filenameslist[0].shape,dtype=tf.int32)
        #这里的filenameslist[1:]被看成一个独立的列表，所以enumerate获取的index跟原来的列表的index相差1
        for index,element in enumerate(filenameslist[1:]):
            labels_Tensor = tf.concat([
                labels_Tensor,
                tf.constant(index+1,shape=element.shape,dtype=tf.int32)],
                axis=-1)
       
        dataset = tf.data.Dataset.from_tensor_slices((filenames_Tensor, labels_Tensor))
        
        #map处理
        dataset = dataset.map(
            map_func=self._decode_and_resize, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=8000)    
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

if __name__ =='__main__':
    loader = dataloader(r'F:\DC_competition\train')
    dataset = loader.load()
    for img,lab in dataset:
        print(lab)