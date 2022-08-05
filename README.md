dataset.py 153 对图像进行了上下裁剪

dataset.py 123 对数据进行了适当均衡，但是扩张不得超过50倍

resnet_csra.py loss_func_lee forward_train_lee forward_test_lee 自定义了loss函数

csra.py 33 对原始输出进行修改，由1个输出头转化为4个输出头（数据集中多标签最大长度为3+背景0）
#### 注：此处为csra创新点，不知道修改的对不对

val.py 93 对4*8 输出头进行第一维(判断为0)进行求均值，取出作为score，从而进行二分类分数进行AUC计算


test.py 为推理代码和test数据的csv，修改19 的backbone 修改24的test img dir 修改25 输出csv地址


 