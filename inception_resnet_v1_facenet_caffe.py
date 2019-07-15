import caffe
from caffe import layers as L
from caffe import params as P

EPS = 0.00001
BN_decay = 0.995 #moving_average_fraction

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

def factorization_conv(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01),
                         bias_filler=dict(type='constant', value=0.0))
    #conv_relu = L.ReLU(conv, in_place=True)

    return conv

def factorization_conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, moving_average_fraction = BN_decay, eps = EPS, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), eps = EPS, in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn,  conv_relu

def factorization_conv_bn_scale_relu_phase(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01))
    conv_bn_tr = L.BatchNorm(conv, use_global_stats=False, moving_average_fraction = BN_decay, in_place=True, eps = EPS, include={'phase':caffe.TRAIN})
    conv_bn = L.BatchNorm(conv, use_global_stats=True, moving_average_fraction = BN_decay, in_place=True, eps = EPS, include={'phase':caffe.TEST})
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn_tr, conv_bn,  conv_relu

def factorization_conv_bn_relu_phase(bottom, num_output=64, kernel_size=3, stride=1, pad=0):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=0.01))
    conv_bn_tr = L.BatchNorm(conv, use_global_stats=False, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TRAIN})
    conv_bn = L.BatchNorm(conv, use_global_stats=True, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TEST})
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn_tr, conv_bn, conv_relu


def factorization_conv_mxn_scale(bottom, num_output=64, kernel_h=1, kernel_w=7, stride=1, pad_h=3, pad_w=0):
    conv_mxn = L.Convolution(bottom, num_output=num_output, kernel_h=kernel_h, kernel_w=kernel_w, stride=stride,
                             pad_h=pad_h, pad_w=pad_w,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', std=0.01))
    conv_mxn_bn_tr = L.BatchNorm(conv_mxn, use_global_stats=False, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TRAIN})
    conv_mxn_bn = L.BatchNorm(conv_mxn, use_global_stats=True, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TEST})
    conv_mxn_scale = L.Scale(conv_mxn, scale_param=dict(bias_term=True), in_place=True)
    conv_mxn_relu = L.ReLU(conv_mxn, in_place=True)

    return conv_mxn, conv_mxn_bn_tr, conv_mxn_bn,  conv_mxn_relu

def factorization_conv_mxn(bottom, num_output=64, kernel_h=1, kernel_w=7, stride=1, pad_h=3, pad_w=0):
    conv_mxn = L.Convolution(bottom, num_output=num_output, kernel_h=kernel_h, kernel_w=kernel_w, stride=stride,
                             pad_h=pad_h, pad_w=pad_w,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', std=0.01))
    conv_mxn_bn_tr = L.BatchNorm(conv_mxn, use_global_stats=False, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TRAIN})
    conv_mxn_bn = L.BatchNorm(conv_mxn, use_global_stats=True, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TEST})
    conv_mxn_relu = L.ReLU(conv_mxn, in_place=True)

    return conv_mxn, conv_mxn_bn_tr, conv_mxn_bn, conv_mxn_relu


def stem_v4_299x299(bottom):
    """
    input:3x160x160
    output:256x17x17
    :param bottom: bottom layer
    :return: layers
    """
    conv1_3x3_s2, conv1_3x3_s2_bn_tr, conv1_3x3_s2_bn,  conv1_3x3_s2_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=32, kernel_size=3, stride=2)  # 32x79x79
    conv2_3x3_s1, conv2_3x3_s1_bn_tr, conv2_3x3_s1_bn,  conv2_3x3_s1_relu = \
        factorization_conv_bn_relu_phase(conv1_3x3_s2, num_output=32, kernel_size=3, stride=1)  # 32x77x77
    conv3_3x3_s1, conv3_3x3_s1_bn_tr, conv3_3x3_s1_bn,  conv3_3x3_s1_relu = \
        factorization_conv_bn_relu_phase(conv2_3x3_s1, num_output=64, kernel_size=3, stride=1, pad=1)  # 64x77x77

    inception_stem1_pool = L.Pooling(conv3_3x3_s1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x38x38
    
    conv4_1x1_s1, conv4_1x1_s1_bn_tr, conv4_1x1_s1_bn,  conv4_1x1_s1_relu = \
        factorization_conv_bn_relu_phase(inception_stem1_pool, num_output=80, kernel_size=1, stride = 1) # 80x38x38
    conv5_3x3_s1, conv5_3x3_s1_bn_tr, conv5_3x3_s1_bn,  conv5_3x3_s1_relu = \
        factorization_conv_bn_relu_phase(conv4_1x1_s1, num_output=192, kernel_size=3, stride=1) # 192x36x36
    conv6_3x3_s2, conv6_3x3_s2_bn_tr, conv6_3x3_s2_bn,  conv6_3x3_s2_relu = \
        factorization_conv_bn_relu_phase(conv5_3x3_s1, num_output=256, kernel_size=3, stride=2) # 256x17x17
    

    return conv1_3x3_s2, conv1_3x3_s2_bn_tr, conv1_3x3_s2_bn,  conv1_3x3_s2_relu, conv2_3x3_s1, conv2_3x3_s1_bn_tr, conv2_3x3_s1_bn, \
            conv2_3x3_s1_relu, conv3_3x3_s1, conv3_3x3_s1_bn_tr, conv3_3x3_s1_bn,  conv3_3x3_s1_relu, \
           inception_stem1_pool, conv4_1x1_s1, conv4_1x1_s1_bn_tr, conv4_1x1_s1_bn,  conv4_1x1_s1_relu, \
           conv5_3x3_s1, conv5_3x3_s1_bn_tr, conv5_3x3_s1_bn,  conv5_3x3_s1_relu, \
           conv6_3x3_s2, conv6_3x3_s2_bn_tr, conv6_3x3_s2_bn,  conv6_3x3_s2_relu

#pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 256x35x35

def inception_v4_a(bottom, n = 256):
    """
    input:256x17x17
    output:256x17x17
    :param bottom: bottom layer
    :return: layers
    """
    
    conv_1x1_b0, conv_1x1_b0_bn_tr, conv_1x1_b0_bn,  conv_1x1_b0_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=32, kernel_size=1)  # 32x17x17

    conv_1x1_b1, conv_1x1_b1_bn_tr, conv_1x1_b1_bn,  conv_1x1_b1_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=32, kernel_size=1)  # 32x17x17
    conv_3x3_b1, conv_3x3_b1_bn_tr, conv_3x3_b1_bn,  conv_3x3_b1_relu = \
        factorization_conv_bn_relu_phase(conv_1x1_b1, num_output=32, kernel_size=3, pad=1)  # 32x17x17

    conv_1x1_b2, conv_1x1_b2_bn_tr, conv_1x1_b2_bn,  conv_1x1_b2_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=32, kernel_size=1)  # 32x17x17
    conv_3x3_b2, conv_3x3_b2_bn_tr, conv_3x3_b2_bn,  conv_3x3_b2_relu = \
        factorization_conv_bn_relu_phase(conv_1x1_b2, num_output=32, kernel_size=3, pad=1)  # 32x17x17
    conv_3x3_b2_a, conv_3x3_b2_a_bn_tr, conv_3x3_b2_a_bn,  conv_3x3_b2_a_relu = \
        factorization_conv_bn_relu_phase(conv_3x3_b2, num_output=32, kernel_size=3, pad=1)  # 32x17x17

    concat = L.Concat(conv_1x1_b0, conv_3x3_b1, conv_3x3_b2_a)  # (32*3=96)x17x17
    conv_1x1 = factorization_conv(concat, num_output=n, kernel_size=1)  
    scaled = L.Power(conv_1x1, power = 1, scale = 0.17, shift = 0) # 256x17x17
    output = L.Eltwise(bottom, scaled)
    output_relu = L.ReLU(output, in_place=True)

    return conv_1x1_b0, conv_1x1_b0_bn_tr, conv_1x1_b0_bn,  conv_1x1_b0_relu, \
           conv_1x1_b1, conv_1x1_b1_bn_tr, conv_1x1_b1_bn,  conv_1x1_b1_relu, \
           conv_3x3_b1, conv_3x3_b1_bn_tr, conv_3x3_b1_bn,  conv_3x3_b1_relu, \
           conv_1x1_b2, conv_1x1_b2_bn_tr, conv_1x1_b2_bn,  conv_1x1_b2_relu, \
           conv_3x3_b2, conv_3x3_b2_bn_tr, conv_3x3_b2_bn,  conv_3x3_b2_relu, \
           conv_3x3_b2_a, conv_3x3_b2_a_bn_tr, conv_3x3_b2_a_bn,  conv_3x3_b2_a_relu, \
           concat, conv_1x1, scaled, output, output_relu


def reduction_v4_a(bottom, k=192, l=192, m=256, n=384):
    """
    input:384x35x35
    output:896x17x17
    :param bottom: bottom layer
    :return: layers
    """

    conv_3x3, conv_3x3_bn_tr, conv_3x3_bn,  conv_3x3_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=n, kernel_size=3, stride=2)  # 384x8x8

    conv_1x1_reduce, conv_1x1_reduce_bn_tr, conv_1x1_reduce_bn,  conv_1x1_reduce_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=k, kernel_size=1)  # 192x17x17
    conv_3x3_2, conv_3x3_2_bn_tr, conv_3x3_2_bn,  conv_3x3_2_relu = \
        factorization_conv_bn_relu_phase(conv_1x1_reduce, num_output=l, kernel_size=3, stride=1, pad =1)  # 192x17x17
    conv_3x3_3, conv_3x3_3_bn_tr, conv_3x3_3_bn,  conv_3x3_3_relu = \
        factorization_conv_bn_relu_phase(conv_3x3_2, num_output=m, kernel_size=3, stride=2)  # 256x8x8

    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 256x8x8

    concat = L.Concat(conv_3x3, conv_3x3_3, pool)  # 896(384+256+256)x8x8

    return conv_3x3, conv_3x3_bn_tr, conv_3x3_bn,  conv_3x3_relu, \
           conv_1x1_reduce, conv_1x1_reduce_bn_tr, conv_1x1_reduce_bn,  conv_1x1_reduce_relu, \
           conv_3x3_2, conv_3x3_2_bn_tr, conv_3x3_2_bn,  conv_3x3_2_relu, \
           conv_3x3_3, conv_3x3_3_bn_tr, conv_3x3_3_bn,  conv_3x3_3_relu, pool, concat


def inception_v4_b(bottom, n = 896):
    """
    input:896x17x17
    output:896x17x17
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn_tr, conv_1x1_bn,  conv_1x1_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=128, kernel_size=1)  # 128x8x8

    conv_1x7_2_reduce, conv_1x7_2_reduce_bn_tr, conv_1x7_2_reduce_bn,  conv_1x7_2_reduce_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=128, kernel_size=1)  # 128x8x8
    conv_1x7_2, conv_1x7_2_bn_tr, conv_1x7_2_bn,  conv_1x7_2_relu = \
        factorization_conv_mxn(conv_1x7_2_reduce, num_output=128, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 128x8x8
    conv_7x1_2, conv_7x1_2_bn_tr, conv_7x1_2_bn,  conv_7x1_2_relu = \
        factorization_conv_mxn(conv_1x7_2, num_output=128, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 128x8x8

    concat = L.Concat(conv_1x1, conv_7x1_2)  # 256(128+128)x8x8
    
    conv_1x1_a = factorization_conv(concat, num_output=n, kernel_size=1)  # 896x8x8
    
    scaled = L.Power(conv_1x1_a, power = 1, scale = 0.10, shift = 0) # 256x17x17
    output = L.Eltwise(bottom, scaled)
    output_relu = L.ReLU(output, in_place=True)

    return conv_1x1, conv_1x1_bn_tr, conv_1x1_bn,  conv_1x1_relu, \
           conv_1x7_2_reduce, conv_1x7_2_reduce_bn_tr, conv_1x7_2_reduce_bn,  conv_1x7_2_reduce_relu, conv_1x7_2, \
           conv_1x7_2_bn_tr, conv_1x7_2_bn,  conv_1x7_2_relu, conv_7x1_2, conv_7x1_2_bn_tr, conv_7x1_2_bn,  \
           conv_7x1_2_relu, concat, conv_1x1_a, scaled, output, output_relu


def reduction_v4_b(bottom):
    """
    input:896x8x8
    output:1792x3x3
    :param bottom: bottom layer
    :return: layers
    """
    
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 896x3x3
    conv_1x1_reduce = factorization_conv(pool, num_output=896, kernel_size=2, stride=1, pad=0)

    conv_3x3_reduce, conv_3x3_reduce_bn_tr, conv_3x3_reduce_bn,  conv_3x3_reduce_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=256, kernel_size=1)  # 256x8x8
    conv_3x3, conv_3x3_bn_tr, conv_3x3_bn,  conv_3x3_relu = \
        factorization_conv_bn_relu_phase(conv_3x3_reduce, num_output=384, kernel_size=3, stride=2)  # 384x3x3

    conv_3x3_reduce_1, conv_3x3_reduce_1_bn_tr, conv_3x3_reduce_1_bn,  conv_3x3_reduce_1_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=256, kernel_size=1)  # 256x8x8
    conv_3x3_1, conv_3x3_1_bn_tr, conv_3x3_1_bn,  conv_3x3_1_relu = \
        factorization_conv_bn_relu_phase(conv_3x3_reduce_1, num_output=256, kernel_size=3, stride=2)  # 256x3x3
    
    conv_3x3_reduce_2, conv_3x3_reduce_2_bn_tr, conv_3x3_reduce_2_bn,  conv_3x3_reduce_2_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=256, kernel_size=1)  # 256x8x8
    conv_3x3_2, conv_3x3_2_bn_tr, conv_3x3_2_bn,  conv_3x3_2_relu = \
        factorization_conv_bn_relu_phase(conv_3x3_reduce_2, num_output=256, kernel_size=3, stride=1, pad = 1)  # 256x8x8
    conv_3x3_2a, conv_3x3_2a_bn_tr, conv_3x3_2a_bn,  conv_3x3_2a_relu = \
        factorization_conv_bn_relu_phase(conv_3x3_2, num_output=256, kernel_size=3, stride=2)  # 256x3x3
    
    concat = L.Concat(conv_3x3, conv_3x3_1, conv_3x3_2a, conv_1x1_reduce)  # 1792(896+256+384+256)x3x3

    return conv_1x1_reduce, pool, conv_3x3_reduce, conv_3x3_reduce_bn_tr, conv_3x3_reduce_bn,  conv_3x3_reduce_relu, \
           conv_3x3, conv_3x3_bn_tr, conv_3x3_bn,  conv_3x3_relu, \
           conv_3x3_reduce_1, conv_3x3_reduce_1_bn_tr, conv_3x3_reduce_1_bn,  conv_3x3_reduce_1_relu, \
           conv_3x3_1, conv_3x3_1_bn_tr, conv_3x3_1_bn,  conv_3x3_1_relu, \
           conv_3x3_reduce_2, conv_3x3_reduce_2_bn_tr, conv_3x3_reduce_2_bn,  conv_3x3_reduce_2_relu, \
           conv_3x3_2, conv_3x3_2_bn_tr, conv_3x3_2_bn,  conv_3x3_2_relu, \
           conv_3x3_2a, conv_3x3_2a_bn_tr, conv_3x3_2a_bn,  conv_3x3_2a_relu, concat
           


def inception_v4_c(bottom, n = 1792, scale = 0.20, activation = True):
    """
    input:1792x3x3
    output:1792x3x3
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1, conv_1x1_bn_tr, conv_1x1_bn,  conv_1x1_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=192, kernel_size=1)  # 192x3x3

    conv_1x1_2, conv_1x1_2_bn_tr, conv_1x1_2_bn,  conv_1x1_2_relu = \
        factorization_conv_bn_relu_phase(bottom, num_output=192, kernel_size=1)  # 192x3x3
    conv_1x3_2, conv_1x3_2_bn_tr, conv_1x3_2_bn,  conv_1x3_2_relu = \
        factorization_conv_mxn(conv_1x1_2, num_output=192, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 192x3x3
    conv_3x1_2, conv_3x1_2_bn_tr, conv_3x1_2_bn,  conv_3x1_2_relu = \
        factorization_conv_mxn(conv_1x3_2, num_output=192, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 192x3x3
    

    concat = L.Concat(conv_1x1, conv_3x1_2)  # 384(192+192)x3x3
    conv_1x1_3 = factorization_conv(concat, num_output=n, kernel_size=1)  # 1792x3x3
    
    scaled = L.Power(conv_1x1_3, power = 1, scale = scale, shift = 0) # 256x17x17
    output = L.Eltwise(bottom, scaled)
    if activation:
        output_relu = L.ReLU(output, in_place=True)
        return conv_1x1, conv_1x1_bn_tr, conv_1x1_bn,  conv_1x1_relu, conv_1x1_2, conv_1x1_2_bn_tr, conv_1x1_2_bn,  conv_1x1_2_relu, conv_1x3_2, conv_1x3_2_bn_tr, conv_1x3_2_bn,  conv_1x3_2_relu, conv_3x1_2, conv_3x1_2_bn_tr, conv_3x1_2_bn,  conv_3x1_2_relu, concat, conv_1x1_3, scaled, output, output_relu
    else:
        return conv_1x1, conv_1x1_bn_tr, conv_1x1_bn,  conv_1x1_relu, conv_1x1_2, conv_1x1_2_bn_tr, conv_1x1_2_bn,  conv_1x1_2_relu, conv_1x3_2, conv_1x3_2_bn_tr, conv_1x3_2_bn,  conv_1x3_2_relu, conv_3x1_2, conv_3x1_2_bn_tr, conv_3x1_2_bn,  conv_3x1_2_relu, concat, conv_1x1_3, scaled, output
        

string_a = 'n.inception_a(order)_1x1_b0, n.inception_a(order)_1x1_b0_bn_tr, n.inception_a(order)_1x1_b0_bn,  n.inception_a(order)_1x1_b0_relu, \
           n.inception_a(order)_1x1_b1, n.inception_a(order)_1x1_b1_bn_tr, n.inception_a(order)_1x1_b1_bn,  n.inception_a(order)_1x1_b1_relu, \
           n.inception_a(order)_3x3_b1, n.inception_a(order)_3x3_b1_bn_tr, n.inception_a(order)_3x3_b1_bn,  n.inception_a(order)_3x3_b1_relu, \
           n.inception_a(order)_1x1_b2, n.inception_a(order)_1x1_b2_bn_tr, n.inception_a(order)_1x1_b2_bn,  n.inception_a(order)_1x1_b2_relu, \
           n.inception_a(order)_3x3_b2, n.inception_a(order)_3x3_b2_bn_tr, n.inception_a(order)_3x3_b2_bn,  n.inception_a(order)_3x3_b2_relu, \
           n.inception_a(order)_3x3_b2_a, n.inception_a(order)_3x3_b2_a_bn_tr, n.inception_a(order)_3x3_b2_a_bn,  n.inception_a(order)_3x3_b2_a_relu, \
           n.inception_a(order)_concat, n.inception_a(order)_1x1, n.inception_a(order)_scaled, n.inception_a(order)_output, n.inception_a(order)_output_relu = \
            inception_v4_a(bottom)'

string_b = 'n.inception_b(order)_1x1, n.inception_b(order)_1x1_bn_tr, n.inception_b(order)_1x1_bn,  n.inception_b(order)_1x1_relu, \
           n.inception_b(order)_1x7_2_reduce, n.inception_b(order)_1x7_2_reduce_bn_tr, n.inception_b(order)_1x7_2_reduce_bn,  n.inception_b(order)_1x7_2_reduce_relu, n.inception_b(order)_1x7_2, \
           n.inception_b(order)_1x7_2_bn_tr, n.inception_b(order)_1x7_2_bn,  n.inception_b(order)_1x7_2_relu, n.inception_b(order)_7x1_2, n.inception_b(order)_7x1_2_bn_tr, n.inception_b(order)_7x1_2_bn,  \
           n.inception_b(order)_7x1_2_relu, n.inception_b(order)_concat, n.inception_b(order)_1x1_a, n.inception_b(order)_scaled, n.inception_b(order)_output, n.inception_b(order)_output_relu = \
            inception_v4_b(bottom)'

string_c = 'n.inception_c(order)_1x1, n.inception_c(order)_1x1_bn_tr, n.inception_c(order)_1x1_bn,  n.inception_c(order)_1x1_relu, \
           n.inception_c(order)_1x1_2, n.inception_c(order)_1x1_2_bn_tr, n.inception_c(order)_1x1_2_bn,  n.inception_c(order)_1x1_2_relu, \
           n.inception_c(order)_1x3_2, n.inception_c(order)_1x3_2_bn_tr, n.inception_c(order)_1x3_2_bn,  n.inception_c(order)_1x3_2_relu, \
           n.inception_c(order)_3x1_2, n.inception_c(order)_3x1_2_bn_tr, n.inception_c(order)_3x1_2_bn,  n.inception_c(order)_3x1_2_relu, \
           n.inception_c(order)_concat, n.inception_c(order)_1x1_3, n.inception_c(order)_scaled, n.inception_c(order)_output, n.inception_c(order)_output_relu = \
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
            source_data_tr = self.train_data
            source_data_ts = self.test_data
            mirror = True
        else:
            source_data = self.test_data
            mirror = False
        n.data, n.label = L.ImageData(name = 'data', batch_size=batch_size, source = source_data_tr, shuffle = mirror,ntop=2,
                                 transform_param=dict(crop_size=256, mean_value=[127.5, 127.5, 127.5], mirror=mirror))

        #n.data, n.label = L.ImageData(name = 'data', batch_size=int(batch_size/2), source = source_data_ts, shuffle = mirror,ntop=2,
        #                         transform_param=dict(crop_size=160, mean_value=[127.5, 127.5, 127.5], mirror=False), include={'phase':caffe.TEST})
                                 
        n.conv1_3x3_s2, n.conv1_3x3_s2_bn_tr, n.conv1_3x3_s2_bn,  n.conv1_3x3_s2_relu, n.conv2_3x3_s1, n.conv2_3x3_s1_bn_tr, n.conv2_3x3_s1_bn, \
            n.conv2_3x3_s1_relu, n.conv3_3x3_s1, n.conv3_3x3_s1_bn_tr, n.conv3_3x3_s1_bn,  n.conv3_3x3_s1_relu, \
           n.inception_stem1_pool, n.conv4_1x1_s1, n.conv4_1x1_s1_bn_tr, n.conv4_1x1_s1_bn,  n.conv4_1x1_s1_relu, \
           n.conv5_3x3_s1, n.conv5_3x3_s1_bn_tr, n.conv5_3x3_s1_bn,  n.conv5_3x3_s1_relu, \
           n.conv6_3x3_s2, n.conv6_3x3_s2_bn_tr, n.conv6_3x3_s2_bn,  n.conv6_3x3_s2_relu = \
           stem_v4_299x299(n.data)  # 384x35x35

        # 5 x inception_a
        for i in range(5):
            if i == 0:
                bottom = 'n.conv6_3x3_s2'
            else:
                bottom = 'n.inception_a(order)_output'.replace('(order)', str(i))
            exec (string_a.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 384x17x17
        
        
        # reduction_v4_a
        n.reduction_a_conv_3x3, n.reduction_a_conv_3x3_bn_tr, n.reduction_a_conv_3x3_bn,  n.reduction_a_conv_3x3_relu, \
        n.reduction_a_conv_1x1_reduce, n.reduction_a_conv_1x1_reduce_bn_tr, n.reduction_a_conv_1x1_reduce_bn,  n.reduction_a_conv_1x1_reduce_relu, \
        n.reduction_a_conv_3x3_2, n.reduction_a_conv_3x3_2_bn_tr, n.reduction_a_conv_3x3_2_bn,  n.reduction_a_conv_3x3_2_relu, \
        n.reduction_a_conv_3x3_3, n.reduction_a_conv_3x3_3_bn_tr, n.reduction_a_conv_3x3_3_bn,  n.reduction_a_conv_3x3_3_relu, n.reduction_a_pool, n.reduction_a_concat = \
        reduction_v4_a(n.inception_a5_output)  # 896x17x17

        # 10 x inception_b
        for i in range(10):
            if i == 0:
                bottom = 'n.reduction_a_concat'
            else:
                bottom = 'n.inception_b(order)_output'.replace('(order)', str(i))
            exec (string_b.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 896x8x8

        # reduction_v4_b
        n.reduction_b_1x1_reduce, n.reduction_b_pool, n.reduction_b_3x3_reduce, n.reduction_b_3x3_reduce_bn_tr, n.reduction_b_3x3_reduce_bn,  n.reduction_b_3x3_reduce_relu, n.reduction_b_3x3, \
           n.reduction_b_3x3_bn_tr, n.reduction_b_3x3_bn,  n.reduction_b_3x3_relu, \
           n.reduction_b_3x3_reduce_1, n.reduction_b_3x3_reduce_1_bn_tr, n.reduction_b_3x3_reduce_1_bn,  n.reduction_b_3x3_reduce_1_relu, \
           n.reduction_b_3x3_1, n.reduction_b_3x3_1_bn_tr, n.reduction_b_3x3_1_bn,  n.reduction_b_3x3_1_relu, \
           n.reduction_b_3x3_reduce_2, n.reduction_b_3x3_reduce_2_bn_tr, n.reduction_b_3x3_reduce_2_bn,  n.reduction_b_3x3_reduce_2_relu, \
           n.reduction_b_3x3_2, n.reduction_b_3x3_2_bn_tr, n.reduction_b_3x3_2_bn,  n.reduction_b_3x3_2_relu, \
           n.reduction_b_3x3_2a, n.reduction_b_3x3_2a_bn_tr, n.reduction_b_3x3_2a_bn,  n.reduction_b_3x3_2a_relu, n.reduction_b_concat = \
            reduction_v4_b(n.inception_b10_output)  # 1792x8x8

        # 5 x inception_c
        for i in range(5):
            if i == 0:
                bottom = 'n.reduction_b_concat'
            else:
                bottom = 'n.inception_c(order)_output'.replace('(order)', str(i))
            exec (string_c.replace('(order)', str(i + 1)).replace('bottom', bottom))  # 1792x3x3
            
        n.Block8_conv_1x1, n.Block8_conv_1x1_bn_tr, n.Block8_conv_1x1_bn,  n.Block8_conv_1x1_relu, n.Block8_conv_1x1_2, \
        n.Block8_conv_1x1_2_bn_tr, n.Block8_conv_1x1_2_bn,  n.Block8_conv_1x1_2_relu, n.Block8_conv_1x3_2, n.Block8_conv_1x3_2_bn_tr, \
        n.Block8_conv_1x3_2_bn,  n.Block8_conv_1x3_2_relu, n.Block8_conv_3x1_2, n.Block8_conv_3x1_2_bn_tr, n.Block8_conv_3x1_2_bn,  \
        n.Block8_conv_3x1_2_relu, n.Block8_concat, n.Block8_conv_1x1_3, n.Block8_scaled, n.Block8_output = \
        inception_v4_c(n.inception_c5_output, scale = 1.0, activation = False)

        n.pool_8x8_s1 = L.Pooling(n.Block8_output, pool=P.Pooling.AVE, global_pooling=True)  # 1792x1x1
        n.pool_8x8_flatten = L.Flatten(n.pool_8x8_s1)
        n.pool_8x8_s1_drop = L.Dropout(n.pool_8x8_flatten, dropout_param=dict(dropout_ratio=0.8))
        n.features = L.InnerProduct(n.pool_8x8_s1_drop, num_output=self.embedding_size,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
        n.features_bn_tr = L.BatchNorm(n.features, use_global_stats=False, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TRAIN})
        n.features_bn = L.BatchNorm(n.features, use_global_stats=True, moving_average_fraction = BN_decay, eps = EPS, in_place=True, include={'phase':caffe.TEST})
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
