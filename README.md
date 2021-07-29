## 使用教程

### 项目目录：

```
/dataset     #数据集存放地
/models      #模型代码
/Output      #输出模型
/tools       #数据加载工具代码
run.ipynb    #运行脚本
SageMaker_run.ipynb       #AWS SageMaker的运行脚本（不用AWS的，可忽略）
SageMaker_run.py	      #AWS SageMaker的运行脚本（不用AWS的，可忽略）
README.md
```



### 使用

下载jupyter notebook，安装tensorflow2.x以上的环境，运行`run.ipynb`即可



### tools介绍

```
h5_to_pb.ipynb     #模型转换工具
tfrec_script.ipynb    #生成.tfrecords文件
dataloader.py   #数据集加载器（加载图片使用）
imgloader.py     #模型转换工具 （图片加载器，主要加载测试集这种不打标签的）
tfrec_pre.py    #.tfrecords文件数据集加载器
```

