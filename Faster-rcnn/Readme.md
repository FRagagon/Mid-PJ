训练步骤
1、数据集的准备
本文使用VOC格式进行训练，训练前需要下载好VOC07的数据集，解压后放在根目录

2、数据集的处理
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。

3、开始网络训练
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。

4、训练结果预测
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。我们首先
要去frcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。
model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。

预测步骤
1、按照训练步骤训练。
2、在frcnn.py文件里面，修改model_path和classes_path使其对应训练好的文件；model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类。
3、运行predict.py，输入检测图片的路径即可，批量预测，可以利用os.listdir()遍历文件夹。

评估步骤
1、在frcnn.py里面修改model_path以及classes_path。model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。
2、运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。
