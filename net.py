import numpy as np
import tensorflow as tf
import numpy as np


def layer(f):
    def _decorator(self, *args, **kwargs):
        assert len(self.inputs)!=0, 'feed it first'
        name = kwargs['name']
        input = self.inputs if len(self.inputs) > 1 else self.inputs[0]
        out = f(self, input, *args, **kwargs)
        self.layers[name] = out
        self.feed(name)
        return self
    return _decorator




n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

network_setting = {'conv_1':[3, 3, 64, 1, 1],
                    'conv_2':[3, 3, 128, 1, 1],
                    'conv_3':[3, 3, 256, 1, 1],
                    'conv_4':[3, 3, 512, 1, 1],
                    'conv_5':[3, 3, 512, 1, 1],
                    'rpn_conv':[3, 3, 512, 1, 1],
                    'rpn_cls_score':[1, 1, len(anchor_scales) * 3 * 2, 1, 1],
                    'pool':[2, 2, 2, 2]

}

class net():
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer,str):
                try:
                    temp = self.layers[layer]
                    self.inputs.append(temp)
                    print('feeding', layer)
                except Exception:
                    print('no layer named {0}'.format(layer))
                    raise Exception()
        return self


    @layer
    def conv(self, input, setting, name, biased=True, relu=True, padding='SAME', trainable=True):

        kernel_hight = setting[0]
        kernel_width = setting[1]
        output_channel = setting[2]
        step_height = setting[3]
        step_width = setting[4]

        print('building',name,'input',input)
        with tf.variable_scope(name) as scope:

            shape = [kernel_hight, kernel_width,input.get_shape()[-1] ,output_channel]
            weigths_init =tf.truncated_normal_initializer(0.0, stddev=0.01)
            weigths = tf.get_variable('weigths', shape, initializer=weigths_init, trainable=trainable)

            if biased:
                biases_init = tf.constant_initializer(0.0)
                biases = tf.get_variable('biases', [output_channel], initializer=biases_init, trainable=trainable)

            con = tf.nn.conv2d(input, weigths, [1, step_height, step_width, 1], padding=padding ,name = 'convolution')

            if biased:
                con = tf.nn.bias_add(con, biases, name = 'bias_add')

            if relu:
                con = tf.nn.relu(con, name= 'relu')

        return con


    @layer
    def max_pool(self, input, setting, name, padding='SAME'):
        return tf.nn.max_pool(input,
                              ksize=[1, setting[0], setting[1], 1],
                              strides=[1, setting[2], setting[3], 1],
                              padding=padding,
                              name=name)

    @layer
    def Bilstm(self, input, setting, name, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)

            img = tf.reshape(input,[shape[0]*shape[1],shape[2],shape[3]])
            img.set_shape([None,None,setting[0]])

            lstm_fw = tf.contrib.rnn.LSTMCell(setting[1], state_is_tuple=True)
            lstm_bw = tf.contrib.rnn.LSTMCell(setting[1], state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, img, dtype=tf.float32)

            lstm_out = tf.concat(lstm_out, axis=-1)
            lstm_out = tf.reshape(lstm_out, [shape[0]*shape[1]*shape[2], 2 * setting[1]])
            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)

            weights = tf.get_variable('weights', [2 * setting[1],setting[2]], initializer = init_weights, trainable = trainable)
            biases = tf.get_variable('biases',[setting[2]], initializer = init_biases, trainable = trainable)

            outputs = tf.matmul(lstm_out,weights)+biases

            outputs = tf.reshape(outputs,[shape[0], shape[1], shape[2], setting[2]])

        return outputs

    @layer
    def lstm_fc(self, input, setting, name, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            input = tf.reshape(input, [shape[0] * shape[1]* shape[2], shape[3]])
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            weights = tf.get_variable('weights', [setting[0], setting[1]], initializer = init_weights, trainable =trainable)
            biases = tf.get_variable('biases', [setting[1]], initializer = init_biases, trainable =trainable)

            output = tf.matmul(input, weights) + biases

            return tf.reshape(output, [shape[0],shape[1], shape[2], setting[1]])


    def _anchor_target_layer(self, rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):


    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):

        a = 1
        with tf.variable_scope(name) as scope:
            tf.py_func



    def setup(self):
        (self.feed('data')
            .conv(network_setting['conv_1'], name = 'conv1_1')
            .conv(network_setting['conv_1'], name = 'conv1_2')
            .max_pool(network_setting['pool'], name = 'max_pooling_1',padding = 'VALID')
            .conv(network_setting['conv_2'], name = 'conv2_1')
            .conv(network_setting['conv_2'], name = 'conv2_2')
            .max_pool(network_setting['pool'], name = 'max_pooling_2',padding = 'VALID')
            .conv(network_setting['conv_3'], name='conv3_1')
            .conv(network_setting['conv_3'], name='conv3_2')
            .conv(network_setting['conv_3'], name='conv3_3')
            .max_pool(network_setting['pool'], name='max_pooling_3', padding='VALID')
            .conv(network_setting['conv_4'], name='conv4_1')
            .conv(network_setting['conv_4'], name='conv4_2')
            .conv(network_setting['conv_4'], name='conv4_3')
            .max_pool(network_setting['pool'], name='max_pooling_4', padding='VALID')
            .conv(network_setting['conv_5'], name='conv5_1')
            .conv(network_setting['conv_5'], name='conv5_2')
            .conv(network_setting['conv_5'], name='conv5_3')
            .conv(network_setting['rpn_conv'], name='rpn_conv_3x3')
            .conv(network_setting['rpn_cls_score'], padding='VALID', relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes','im_info','data'))
        print(self.layers['rpn_cls_score'])



#network = net('a')
#network.feed('data').Bilstm(network_setting['conv_1'],name = 'ddd')


#print((network.layers['data'].get_shape()[-1]))
a = [[1,2,3,4,5]]
b = np.array(a)-1
print(b.shape[0])