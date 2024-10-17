# Cross-modal Feature Fusion via Mutual
Assistance: A Novel Network for Enhanced
Object Detection

## 环境要求
Python>=3.6.0 is required with all requirements.txt installed including PyTorch>=1.7 (The same as yolov5 https://github.com/ultralytics/yolov5 ).


  
#### 环境安装


pip install -r requirements.txt


## 数据集


-[LLVIP]  [download](https://github.com/bupt-ai-cz/LLVIP)
此数据集为暗光条件下的行人数据集，只包含行人这一类别。数据集自带划分细节，包含训练集和测试集（无验证集）
-[M3FD]  [download]
此数据集为交通道路数据集包含行人、摩托车、汽车、交通信号灯、货车、公交车等类别。由于此数据集无官方划分结果，该数据集划分参考该文章：
Mingjian Liang, Junjie Hu, Chenyu Bao, Hua Feng, Fuqin
Deng, and Tin Lun Lam. Explicit attention-enhanced fu-
sion for rgb-thermal perception tasks. IEEE Robotics Autom.
Lett., 8(7):4060–4067, 2023. 
至于其他方法在此数据集上的结果，请参考Fusion-Mamba for Cross-modality Object Detection这篇文章，同时本论文方法也与该文章方法进行了比较。

此外，所有数据集的标注都应转换为yolov5格式

参考: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

##训练
#### 下载yolov5预训练权重
yolov5 weights (pre-train) 

-[yolov5s] [google drive](https://drive.google.com/file/d/1UGAsaOvV7jVrk0RvFVYL6Vq0K7NQLD8H/view?usp=sharing)

-[yolov5m] [google drive](https://drive.google.com/file/d/1qB7L2vtlGppGjHp5xpXCKw14YHhbV4s1/view?usp=sharing)

-[yolov5l] [google drive](https://drive.google.com/file/d/12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb/view?usp=sharing)

-[yolov5x] [google drive](https://drive.google.com/file/d/1e9xiQImx84KFQ_a7XXpn608I3rhRmKEn/view?usp=sharing)



#### 更改数据集文件地址
".\data\multispectral"
可通过whereimages.py文件设置存放图片路径的txt文档，即train.txt val.txt和test.txt
数据集具体存放位置如下（以M3FD为例）：
-M3FD
	-visible
		-train
			-labels
			-images
		-val
			-labels
			-images
		-test
			-labels
			-images
		train.txt
		val.txt
		test.txt
	-infrared
		-train
			-labels
			-images
		-val
			-labels
			-images
		-test
			-labels
			-images
		train.txt
		val.txt
		test.txt


执行指令：``` python train.py```

#### 测试

训练权重：

通过网盘分享的文件：training weights.zip
链接: https://pan.baidu.com/s/14MahN3Hi3yDgiQ-GnU3EEg 提取码: u4dm

我们提供了详尽的训练权重和所有的训练细节，以供读者进行测试。
在我们提供的网盘中一共有三个文件夹分别是detect、test 和 train。
detect用于存放两个数据集在网络训练测试后的可视化结果，里边一共有两个文件夹分别为两个数据集的检测结果可视化。
test用于存储训练后进行测试后每张图的结果
train用于存储每次训练后得到的信息，包括权重、每一轮的验证结果、loss图、和其他一些训练日志相关信息。

加载好训练权重、配置好数据集位置后权重输入指令 ``` python test.py```即可开始进行验证工作。

若要通过训练权重进行可视化输入指令``` python detect_twostream.py```






