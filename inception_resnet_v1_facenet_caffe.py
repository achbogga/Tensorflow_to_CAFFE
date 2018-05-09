import caffe
from caffe import layers as L
from caffe import params as P


def l2normed(vec, dim):
    """Returns L2-normalized instances of vec; i.e., for each instance x in vec,
    computes  x / ((x ** 2).sum() ** 0.5). Assumes vec has shape N x dim."""
    denom = L.Reduction(vec, axis=1, operation=P.Reduction.SUMSQ)
    denom = L.Power(denom, power=(-0.5), shift=1e-12)
    denom = L.Reshape(denom, num_axes=0, axis=-1, shape=dict(dim=[1]))
    denom = L.Tile(denom, axis=1, tiles=dim)
    return L.Eltwise(vec, denom, operation=P.Eltwise.PROD)

def fc_relu_drop(bottom, num_output=1024, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=num_output,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='xavier', std=1),
                        bias_filler=dict(type='constant', value=0.2))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def factorization_conv_mxn(bottom, num_output=64, kernel_h=1, kernel_w=7, stride=1, pad_h=3, pad_w=0):
    conv_mxn = L.Convolution(bottom, num_output=num_output, kernel_h=kernel_h, kernel_w=kernel_w, stride=stride,
                             pad_h=pad_h, pad_w=pad_w,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', std=0.01),
                             bias_filler=dict(type='constant', value=0.2))
    conv_mxn_bn = L.BatchNorm(conv_mxn, use_global_stats=False, in_place=True)
    conv_mxn_scale = L.Scale(conv_mxn, scale_param=dict(bias_term=True), in_place=True)
    conv_mxn_relu = L.ReLU(conv_mxn, in_place=True)

    return conv_mxn, conv_mxn_bn, conv_mxn_scale, conv_mxn_relu


def stem_v4_299x299(bottom):
    """
    input:3x299x299
    output:256x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv1_3x3_s2, conv1_3x3_s2_bn, conv1_3x3_s2_scale, conv1_3x3_s2_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=3, stride=2)  # 32x149x149
    conv2_3x3_s1, conv2_3x3_s1_bn, conv2_3x3_s1_scale, conv2_3x3_s1_relu = \
        factorization_conv_bn_scale_relu(conv1_3x3_s2, num_output=32, kernel_size=3, stride=1)  # 32x147x147
    conv3_3x3_s1, conv3_3x3_s1_bn, conv3_3x3_s1_scale, conv3_3x3_s1_relu = \
        factorization_conv_bn_scale_relu(conv2_3x3_s1, num_output=64, kernel_size=3, stride=1, pad=1)  # 64x147x147

    inception_stem1_pool = L.Pooling(conv3_3x3_s1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x73x73
    
    conv4_1x1_s1, conv4_1x1_s1_bn, conv4_1x1_s1_scale, conv4_1x1_s1_relu = \
        factorization_conv_bn_scale_relu(conv3_3x3_s1, num_output=80, kernel_size=1, stride = 1)
    conv5_3x3_s1, conv5_3x3_s1_bn, conv5_3x3_s1_scale, conv5_3x3_s1_relu = \
        factorization_conv_bn_scale_relu(conv4_1x1_s1, num_output=192, kernel_size=3, stride=1)
    conv6_3x3_s2, conv6_3x3_s2_bn, conv6_3x3_s2_scale, conv6_3x3_s2_relu = \
        factorization_conv_bn_scale_relu(conv5_3x3_s1, num_output=256, kernel_size=3, stride=2)
    

    return conv1_3x3_s2, conv1_3x3_s2_bn, conv1_3x3_s2_scale, conv1_3x3_s2_relu, conv2_3x3_s1, conv2_3x3_s1_bn, \
           conv2_3x3_s1_scale, conv2_3x3_s1_relu, conv3_3x3_s1, conv3_3x3_s1_bn, conv3_3x3_s1_scale, conv3_3x3_s1_relu, \
           inception_stem1_pool, conv4_1x1_s1, conv4_1x1_s1_bn, conv4_1x1_s1_scale, conv4_1x1_s1_relu, \
           conv5_3x3_s1, conv5_3x3_s1_bn, conv5_3x3_s1_scale, conv5_3x3_s1_relu, \
           conv6_3x3_s2, conv6_3x3_s2_bn, conv6_3x3_s2_scale, conv6_3x3_s2_relu

#pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 256x35x35

def inception_v4_a(bottom, n = 128):
    """
    input:256x35x35
    output:256x35x35
    :param bottom: bottom layer
    :return: layers
    """
    
    conv_1x1_b0, conv_1x1_b0_bn, conv_1x1_b0_scale, conv_1x1_b0_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=1)  # 32x35x35

    conv_1x1_b1, conv_1x1_b1_bn, conv_1x1_b1_scale, conv_1x1_b1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=1)  # 32x35x35
    conv_3x3_b1, conv_3x3_b1_bn, conv_3x3_b1_scale, conv_3x3_b1_relu = \
        factorization_conv_bn_scale_relu(conv_1x1_b1, num_output=32, kernel_size=3, pad=1)  # 32x35x35

    conv_1x1_b2, conv_1x1_b2_bn, conv_1x1_b2_scale, conv_1x1_b2_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=32, kernel_size=1)  # 32x35x35
    conv_3x3_b2, conv_3x3_b2_bn, conv_3x3_b2_scale, conv_3x3_b2_relu = \
        factorization_conv_bn_scale_relu(conv_1x1_b2, num_output=32, kernel_size=3, pad=1)  # 32x35x35
    conv_3x3_b2_a, conv_3x3_b2_a_bn, conv_3x3_b2_a_scale, conv_3x3_b2_a_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_b2, num_output=32, kernel_size=3, pad=1)  # 32x35x35

    concat = L.Concat(conv_1x1_b0, conv_3x3_b1, conv_3x3_b2_a)  # (32*3=96)x35x35
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(concat, num_output=n, kernel_size=1)  # 256x35x35
    

    return conv_1x1_b0, conv_1x1_b0_bn, conv_1x1_b0_scale, conv_1x1_b0_relu, \
           conv_1x1_b1, conv_1x1_b1_bn, conv_1x1_b1_scale, conv_1x1_b1_relu, \
           conv_3x3_b1, conv_3x3_b1_bn, conv_3x3_b1_scale, conv_3x3_b1_relu, \
           conv_1x1_b2, conv_1x1_b2_bn, conv_1x1_b2_scale, conv_1x1_b2_relu, \
           conv_3x3_b2, conv_3x3_b2_bn, conv_3x3_b2_scale, conv_3x3_b2_relu, \
           conv_3x3_b2_a, conv_3x3_b2_a_bn, conv_3x3_b2_a_scale, conv_3x3_b2_a_relu, \
           concat, conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu


def reduction_v4_a(bottom, k=192, l=192, m=256, n=384):
    """
    input:384x35x35
    output:1024x17x17
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 256x17x17

    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=n, kernel_size=3, stride=2)  # 256x17x17

    conv_3x3_2_reduce, conv_3x3_2_reduce_bn, conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=k, kernel_size=1)  # 192x35x35
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2_reduce, num_output=l, kernel_size=3, stride=1, pad=1)  # 192x35x35
    conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2, num_output=m, kernel_size=3, stride=2)  # 256x17x17

    concat = L.Concat(pool, conv_3x3, conv_3x3_3)  # 768(256+256+256)x17x17

    return pool, conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, conv_3x3_2_reduce, conv_3x3_2_reduce_bn, \
           conv_3x3_2_reduce_scale, conv_3x3_2_reduce_relu, conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, \
           conv_3x3_2_relu, conv_3x3_3, conv_3x3_3_bn, conv_3x3_3_scale, conv_3x3_3_relu, concat


def inception_v4_b(bottom, n = 448):
    """
    input:1024x17x17
    output:1024x17x17
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=128, kernel_size=1)  # 128x17x17

    conv_1x7_2_reduce, conv_1x7_2_reduce_bn, conv_1x7_2_reduce_scale, conv_1x7_2_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=128, kernel_size=1)  # 128x17x17
    conv_1x7_2, conv_1x7_2_bn, conv_1x7_2_scale, conv_1x7_2_relu = \
        factorization_conv_mxn(conv_1x7_2_reduce, num_output=128, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 128x17x17
    conv_7x1_2, conv_7x1_2_bn, conv_7x1_2_scale, conv_7x1_2_relu = \
        factorization_conv_mxn(conv_1x7_2, num_output=128, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 128x17x17

    concat = L.Concat(conv_1x1, conv_7x1_2)  # 256(128+128)x17x17
    
    conv_1x1_a, conv_1x1_a_bn, conv_1x1_a_scale, conv_1x1_a_relu = \
        factorization_conv_bn_scale_relu(concat, num_output=n, kernel_size=1)  # 896x17x17

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, \
           conv_1x7_2_reduce, conv_1x7_2_reduce_bn, conv_1x7_2_reduce_scale, conv_1x7_2_reduce_relu, conv_1x7_2, \
           conv_1x7_2_bn, conv_1x7_2_scale, conv_1x7_2_relu, conv_7x1_2, conv_7x1_2_bn, conv_7x1_2_scale, \
           conv_7x1_2_relu, concat, conv_1x1_a, conv_1x1_a_bn, conv_1x1_a_scale, conv_1x1_a_relu


def reduction_v4_b(bottom):
    """
    input:1024x17x17
    output:1536x8x8
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX, ceil_mode = False)  # 768x8x8

    conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x17x17
    conv_3x3, conv_3x3_bn, conv_3x3_scale, conv_3x3_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce, num_output=384, kernel_size=3, stride=2)  # 256x8x8

    conv_3x3_reduce_1, conv_3x3_reduce_1_bn, conv_3x3_reduce_1_scale, conv_3x3_reduce_1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x17x17
    conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_scale, conv_3x3_1_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce_1, num_output=256, kernel_size=3, stride=2)  # 256x8x8
    
    conv_3x3_reduce_2, conv_3x3_reduce_2_bn, conv_3x3_reduce_2_scale, conv_3x3_reduce_2_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=256, kernel_size=1)  # 256x17x17
    conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_reduce_2, num_output=256, kernel_size=3, stride=1)  # 256x8x8
    conv_3x3_2a, conv_3x3_2a_bn, conv_3x3_2a_scale, conv_3x3_2a_relu = \
        factorization_conv_bn_scale_relu(conv_3x3_2, num_output=256, kernel_size=3, stride=2, pad =1)  # 256x8x8

    concat = L.Concat(pool, conv_3x3, conv_3x3_1, conv_3x3_2a)  # 1920(1024+256+384+256)x8x8

    return pool, conv_3x3_reduce, conv_3x3_reduce_bn, conv_3x3_reduce_scale, conv_3x3_reduce_relu, conv_3x3, \
           conv_3x3_bn, conv_3x3_scale, conv_3x3_relu, \
           conv_3x3_reduce_1, conv_3x3_reduce_1_bn, conv_3x3_reduce_1_scale, conv_3x3_reduce_1_relu, \
           conv_3x3_1, conv_3x3_1_bn, conv_3x3_1_scale, conv_3x3_1_relu, \
           conv_3x3_reduce_2, conv_3x3_reduce_2_bn, conv_3x3_reduce_2_scale, conv_3x3_reduce_2_relu, \
           conv_3x3_2, conv_3x3_2_bn, conv_3x3_2_scale, conv_3x3_2_relu, \
           conv_3x3_2a, conv_3x3_2a_bn, conv_3x3_2a_scale, conv_3x3_2a_relu, concat
           


def inception_v4_c(bottom, n = 896):
    """
    input:1536x8x8
    output:1536x8x8
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x8x8

    conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, conv_1x1_2_relu = \
        factorization_conv_bn_scale_relu(bottom, num_output=192, kernel_size=1)  # 192x8x8
    conv_1x3_2, conv_1x3_2_bn, conv_1x3_2_scale, conv_1x3_2_relu = \
        factorization_conv_mxn(conv_1x1_2, num_output=192, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 192x8x8
    conv_3x1_2, conv_3x1_2_bn, conv_3x1_2_scale, conv_3x1_2_relu = \
        factorization_conv_mxn(conv_1x3_2, num_output=192, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 192x8x8
    

    concat = L.Concat(conv_1x1, conv_3x1_2)  # 384(192+192)x17x17
    conv_1x1_3, conv_1x1_3_bn, conv_1x1_3_scale, conv_1x1_3_relu = \
        factorization_conv_bn_scale_relu(concat, num_output=n, kernel_size=1)  # 1792x8x8

    return conv_1x1, conv_1x1_bn, conv_1x1_scale, conv_1x1_relu, \
           conv_1x1_2, conv_1x1_2_bn, conv_1x1_2_scale, conv_1x1_2_relu, \
           conv_1x3_2, conv_1x3_2_bn, conv_1x3_2_scale, conv_1x3_2_relu, \
           conv_3x1_2, conv_3x1_2_bn, conv_3x1_2_scale, conv_3x1_2_relu, \
           concat, conv_1x1_3, conv_1x1_3_bn, conv_1x1_3_scale, conv_1x1_3_relu


string_a = 'n.inception_a(order)_1x1_b0, n.inception_a(order)_1x1_b0_bn, n.inception_a(order)_1x1_b0_scale, n.inception_a(order)_1x1_b0_relu, \
           n.inception_a(order)_1x1_b1, n.inception_a(order)_1x1_b1_bn, n.inception_a(order)_1x1_b1_scale, n.inception_a(order)_1x1_b1_relu, \
           n.inception_a(order)_3x3_b1, n.inception_a(order)_3x3_b1_bn, n.inception_a(order)_3x3_b1_scale, n.inception_a(order)_3x3_b1_relu, \
           n.inception_a(order)_1x1_b2, n.inception_a(order)_1x1_b2_bn, n.inception_a(order)_1x1_b2_scale, n.inception_a(order)_1x1_b2_relu, \
           n.inception_a(order)_3x3_b2, n.inception_a(order)_3x3_b2_bn, n.inception_a(order)_3x3_b2_scale, n.inception_a(order)_3x3_b2_relu, \
           n.inception_a(order)_3x3_b2_a, n.inception_a(order)_3x3_b2_a_bn, n.inception_a(order)_3x3_b2_a_scale, n.inception_a(order)_3x3_b2_a_relu, \
           n.inception_a(order)_concat, n.inception_a(order)_1x1, n.inception_a(order)_1x1_bn, n.inception_a(order)_1x1_scale, n.inception_a(order)_1x1_relu = \
            inception_v4_a(bottom)'

string_b = 'n.inception_b(order)_1x1, n.inception_b(order)_1x1_bn, n.inception_b(order)_1x1_scale, n.inception_b(order)_1x1_relu, \
           n.inception_b(order)_1x7_2_reduce, n.inception_b(order)_1x7_2_reduce_bn, n.inception_b(order)_1x7_2_reduce_scale, n.inception_b(order)_1x7_2_reduce_relu, n.inception_b(order)_1x7_2, \
           n.inception_b(order)_1x7_2_bn, n.inception_b(order)_1x7_2_scale, n.inception_b(order)_1x7_2_relu, n.inception_b(order)_7x1_2, n.inception_b(order)_7x1_2_bn, n.inception_b(order)_7x1_2_scale, \
           n.inception_b(order)_7x1_2_relu, n.inception_b(order)_concat, n.inception_b(order)_1x1_a, n.inception_b(order)_1x1_a_bn, n.inception_b(order)_1x1_a_scale, n.inception_b(order)_1x1_a_relu = \
            inception_v4_b(bottom)'

string_c = 'n.inception_c(order)_1x1, n.inception_c(order)_1x1_bn, n.inception_c(order)_1x1_scale, n.inception_c(order)_1x1_relu, \
           n.inception_c(order)_1x1_2, n.inception_c(order)_1x1_2_bn, n.inception_c(order)_1x1_2_scale, n.inception_c(order)_1x1_2_relu, \
           n.inception_c(order)_1x3_2, n.inception_c(order)_1x3_2_bn, n.inception_c(order)_1x3_2_scale, n.inception_c(order)_1x3_2_relu, \
           n.inception_c(order)_3x1_2, n.inception_c(order)_3x1_2_bn, n.inception_c(order)_3x1_2_scale, n.inception_c(order)_3x1_2_relu, \
           n.inception_c(order)_concat, n.inception_c(order)_1x1_3, n.inception_c(order)_1x1_3_bn, n.inception_c(order)_1x1_3_scale, n.inception_c(order)_1x1_3_relu = \
            inception_v4_c(bottom)'


class InceptionV4(object):
    def __init__(self, train_src = 'train.txt', test_src = 'val.txt', embedding_size = 512, num_output = 9131):
        self.train_data = train_src
        self.test_data = test_src
        self.classifier_num = num_output
        self.embedding_size = embedding_size

    def inception_v4_proto(self, batch_size, phase='TRAIN'):
        n = caffe.NetSpec()
        if phase == 'TRAIN':
            source_data = self.train_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.ImageData(name = 'data', batch_size=batch_size, source = source_data, shuffle = mirror,ntop=2,
                                 transform_param=dict(crop_size=160, mean_value=[127.5, 127.5, 127.5], mirror=mirror))

        
        n.conv1_3x3_s2, n.conv1_3x3_s2_bn, n.conv1_3x3_s2_scale, n.conv1_3x3_s2_relu, n.conv2_3x3_s1, n.conv2_3x3_s1_bn, \
        n.conv2_3x3_s1_scale, n.conv2_3x3_s1_relu, n.conv3_3x3_s1, n.conv3_3x3_s1_bn, n.conv3_3x3_s1_scale, n.conv3_3x3_s1_relu, \
        inception_stem1_pool, n.conv4_1x1_s1, n.conv4_1x1_s1_bn, n.conv4_1x1_s1_scale, n.conv4_1x1_s1_relu, \
        n.conv5_3x3_s1, n.conv5_3x3_s1_bn, n.conv5_3x3_s1_scale, n.conv5_3x3_s1_relu, \
        n.conv6_3x3_s2, n.conv6_3x3_s2_bn, n.conv6_3x3_s2_scale, n.conv6_3x3_s2_relu = \
           stem_v4_299x299(n.data)  # 384x35x35

        # 5 x inception_a
        for i in range(5):
            if i == 0:
                bottom = 'n.conv6_3x3_s2'
            else:
                bottom = 'n.inception_a(order)_1x1'.replace('(order)', str(i))
            exec (string_a.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 384x35x35

        # reduction_v4_a
        n.reduction_a_pool, n.reduction_a_3x3, n.reduction_a_3x3_bn, n.reduction_a_3x3_scale, n.reduction_a_3x3_relu, n.reduction_a_3x3_2_reduce, n.reduction_a_3x3_2_reduce_bn, \
        n.reduction_a_3x3_2_reduce_scale, n.reduction_a_3x3_2_reduce_relu, n.reduction_a_3x3_2, n.reduction_a_3x3_2_bn, n.reduction_a_3x3_2_scale, \
        n.reduction_a_3x3_2_relu, n.reduction_a_3x3_3, n.reduction_a_3x3_3_bn, n.reduction_a_3x3_3_scale, n.reduction_a_3x3_3_relu, n.reduction_a_concat = \
        reduction_v4_a(n.inception_a5_1x1)  # 1024x17x17

        # 10 x inception_b
        for i in range(10):
            if i == 0:
                bottom = 'n.reduction_a_concat'
            else:
                bottom = 'n.inception_b(order)_1x1_a'.replace('(order)', str(i))
            exec (string_b.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 1024x17x17

        # reduction_v4_b
        n.reduction_b_pool, n.reduction_b_3x3_reduce, n.reduction_b_3x3_reduce_bn, n.reduction_b_3x3_reduce_scale, n.reduction_b_3x3_reduce_relu, n.reduction_b_3x3, \
           n.reduction_b_3x3_bn, n.reduction_b_3x3_scale, n.reduction_b_3x3_relu, \
           n.reduction_b_3x3_reduce_1, n.reduction_b_3x3_reduce_1_bn, n.reduction_b_3x3_reduce_1_scale, n.reduction_b_3x3_reduce_1_relu, \
           n.reduction_b_3x3_1, n.reduction_b_3x3_1_bn, n.reduction_b_3x3_1_scale, n.reduction_b_3x3_1_relu, \
           n.reduction_b_3x3_reduce_2, n.reduction_b_3x3_reduce_2_bn, n.reduction_b_3x3_reduce_2_scale, n.reduction_b_3x3_reduce_2_relu, \
           n.reduction_b_3x3_2, n.reduction_b_3x3_2_bn, n.reduction_b_3x3_2_scale, n.reduction_b_3x3_2_relu, \
           n.reduction_b_3x3_2a, n.reduction_b_3x3_2a_bn, n.reduction_b_3x3_2a_scale, n.reduction_b_3x3_2a_relu, n.reduction_b_concat = \
            reduction_v4_b(n.inception_b10_1x1_a)  # 1536x8x8

        # 5 x inception_c
        for i in range(5):
            if i == 0:
                bottom = 'n.reduction_b_concat'
            else:
                bottom = 'n.inception_c(order)_1x1_3'.replace('(order)', str(i))
            exec (string_c.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 1792x8x8

        n.pool_8x8_s1 = L.Pooling(n.inception_c5_1x1_3, pool=P.Pooling.AVE, global_pooling=True)  # 1792x1x1
        n.pool_8x8_s1_drop = L.Dropout(n.pool_8x8_s1, dropout_param=dict(dropout_ratio=0.2))
        n.features = L.InnerProduct(n.pool_8x8_s1_drop, num_output=self.embedding_size,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
        n.normed_features = l2normed(n.features, self.embedding_size)
        n.classifier = L.InnerProduct(n.normed_features, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
        
        n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        if phase == 'TRAIN':
            pass
        else:
            n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
            n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),
                                         accuracy_param=dict(top_k=5))

        return n.to_proto()
