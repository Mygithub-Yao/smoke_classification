## dataset文件夹解析

dataset主要分为：

- test  （存储测试使用的图片）
- train   （按分类创建文件夹）

### .tfrecords文件

是tensorflow一种存储数据的结构化文件，处理过后的数据集将保持在.tfrecords文件中，调用tools中的代码即可生成。