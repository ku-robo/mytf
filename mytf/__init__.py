import tensorflow as tf


class Mytf:
    sess = tf.Session()
    keep_prob = tf.placeholder(tf.float32)
    prediction = []
    Optimizer = None
    train_step = None

    def __init__(self, in_size, out_size):
        print('新建神经网络, 特征数量', in_size, ', 结果类型数量', out_size)
        self.prediction.append(tf.placeholder(tf.float32, [None, in_size]))
        self.train_out = tf.placeholder(tf.float32, [None, out_size])

    # 添加 卷积 层
    # patch_size 卷积核大小
    # step_size 移动步长
    # step_size <= patch_size 步长要小于等于核大小,否则有扫不到的地方
    def add_conv_layer(self, patch_size, step_size, out_size, padding='SAME', ac_fun=tf.nn.relu):
        in_size = int(self.prediction[-1].shape[-1])
        weight = self.variable_weight([patch_size, patch_size, in_size, out_size])
        bias = self.variable_bias([out_size])
        # input         输入图像                  [batch, in_height, in_width, in_channels]
        # filter/weight 卷积核                    [filter_height, filter_width, in_channels, out_channels]
        # strides       卷积核每一步移动长度步长     [batch?, x方向步长, y方向步长, 1通道?]
        # padding       VALID保证全部在图片内部,宽度不够丢弃,可能小一圈 SAME步长超过图片补零,保证结果和原图一样大
        conv = tf.nn.conv2d(self.prediction[-1], weight, strides=[1, step_size, step_size, 1], padding=padding) + bias
        self.prediction.append(ac_fun(conv))
        print('添加卷积层', self.prediction[-2].shape, '=>', self.prediction[-1].shape,
              '\t卷积核大小{patch_size}\t移动步长{step_size}'.format(**locals()))

    # 添加 池化 层
    # pool_size 池化窗口大小
    # step_size 移动步长
    # 一般pool_size == step_size
    def add_pool_layer(self, pool_size, step_size, padding='SAME'):
        after_pool = tf.nn.max_pool(self.prediction[-1], ksize=[1, pool_size, pool_size, 1],
                                    strides=[1, step_size, step_size, 1], padding=padding)
        self.prediction.append(after_pool)
        print('添加池化层', self.prediction[-2].shape, '=>', self.prediction[-1].shape)

    # fully connected layer
    def add_fc_layer(self, out_size, ac_fun=lambda f: f):
        in_size = int(self.prediction[-1].shape[-1])
        weight = self.variable_weight([in_size, out_size])
        bias = self.variable_bias([out_size])
        after_matmul = tf.matmul(self.prediction[-1], weight) + bias
        self.prediction.append(ac_fun(after_matmul))
        print('添加神经层', self.prediction[-2].shape, '=>', self.prediction[-1].shape)

    def reshape_fc(self):
        wd = 1
        for _ in self.prediction[-1].shape[1:]:
            wd *= _
        self.reshape([-1, wd])

    def reshape(self, shape):
        after_reshape = tf.reshape(self.prediction[-1], shape)
        self.prediction.append(after_reshape)
        print('数据变形', self.prediction[-2].shape, '=>', self.prediction[-1].shape)

    def dropout(self):
        print('随机丢弃神经元')
        after_dropout = tf.nn.dropout(self.prediction[-1], self.keep_prob)
        self.prediction.append(after_dropout)

    # 返回 新生成权重 变量, 输入形状 初始随机分布0.1
    @staticmethod
    def variable_weight(shape, stddev=0.1):
        return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

        # 返回 新生成偏置 变量 输入形状 默认全为0.1

    @staticmethod
    def variable_bias(shape, default=0.1):
        return tf.Variable(tf.constant(default, shape=shape))

    # 最后循环跑这个
    def run_cnn(self, train_in, train_out, keep_prob):
        if not self.Optimizer:
            self.Optimizer = tf.train.AdamOptimizer(1e-4).minimize(
                tf.reduce_mean(-tf.reduce_sum(self.train_out * tf.log(self.prediction[-1]), reduction_indices=[1]))
            )
            self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.Optimizer, feed_dict={
            self.prediction[0]: train_in,
            self.train_out: train_out,
            self.keep_prob: keep_prob
        })

    # 用模型计算测试输出,返回准确率
    def test(self, test_in, test_out):
        # 用测试数据输入,得到基于训练的测试输出结果 结果是 hot-space 形式
        train_test_out = self.sess.run(self.prediction[-1],
                                       feed_dict={self.prediction[0]: test_in, self.keep_prob: 1})
        # 用 基于训练的测试输出结果 和真实测试输出 对比 得到真实预测 bool类型
        correct_prediction = tf.equal(tf.argmax(train_test_out, 1), tf.argmax(test_out, 1))
        # 转化 bool 类型结果为 float 然后取平均数
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 运算得到 accuracy 的值 不要'calc_out': v_ys可不可以??? self.prediction[0]: test_in,不要可不可以???
        result = self.sess.run(accuracy)
        return result


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 定义输入特征数量784,输出特征数量10
    mytf = Mytf(784, 10)
    # 输入的[None,784],把784增加到维度上,分别是n行,28*28像素*1通道颜色
    mytf.reshape([-1, 28, 28, 1])
    # 卷积 图片尺寸不变, 特征层数从1到32
    mytf.add_conv_layer(5, 1, 32)
    # 池化 图片尺寸从 28*28 变为 14*14
    mytf.add_pool_layer(2, 2)
    # 卷积 图片尺寸不变, 特征层数从32到64
    mytf.add_conv_layer(5, 1, 64)
    # 池化 图片尺寸从 14*14 变为 7*7
    mytf.add_pool_layer(2, 2)

    # 维度恢复到2维,为全连接提取特征, 尺寸为 n行* 7*7像素* 64层特征
    mytf.reshape_fc()
    # 隐藏层 从7 * 7 * 64特征数减少到 1024
    mytf.add_fc_layer(1024, tf.nn.relu)
    # 随机丢弃部分神经元
    mytf.dropout()
    # 隐藏层 从1024特征数减少到 10
    mytf.add_fc_layer(10, tf.nn.softmax)

    for i in range(1000):
        _train_in1, _train_out = mnist.train.next_batch(100)
        mytf.run_cnn(_train_in1, _train_out, 0.5)
        if i % 20 == 0:
            print(mytf.test(mnist.test.images[:1000], mnist.test.labels[:1000]))
