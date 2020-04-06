# 量化相关基础函数

import tensorflow as tf


def compute_quant_param(data, bitnum, ifscale=False, ifoffset=False, ifchannel=False):
    """
    计算量化参数
    :param data: 需要被量化的数据
    :param bitnum: 量化位宽
    :param ifscale: 是否缩放
    :param ifoffset: 是否为非对称成量化
    :param ifchannel: 是否分通道
    :return: 量化参数shift、scale、offset
    """
    with tf.variable_scope(data.name[:-2] + "/quantize", reuse=tf.AUTO_REUSE):
        axis = (0, 1, 2) if ifchannel else None
        if ifoffset:
            data_max = tf.reduce_max(data, axis=axis)
            data_min = tf.reduce_min(data, axis=axis)
        else:
            data_max = tf.reduce_max(tf.abs(data), axis=axis)
            data_min = -data_max
        Z = (data_max - data_min) / 2 + 1e-9
        o = (data_max + data_min) / 2
        shift = tf.ceil(tf.log(Z / (2 ** (bitnum - 1) - 1)) / tf.log(2))
        f = Z / (tf.pow(2.0, shift) * (tf.pow(2.0, (bitnum - 1)) - 1)) if ifscale else tf.ones_like(shift)
        return shift, f, o


def float2fix(data, shift, f, o):
    """
    根据量化参数将数据量化为int，再反量化成float
    :param data: 需要量化的数据
    :param shift: shift
    :param f: scale
    :param o: offset
    :return: 通过量化处理的数据
    """
    with tf.variable_scope(data.name[:-2] + "/dequantize", reuse=tf.AUTO_REUSE):
        # 量化成int
        tmp1 = (data - o) / (tf.pow(2.0, shift) * f)
        tmp2 = tf.round(tmp1)

        # 反量化
        fix = tmp2 * tf.pow(2.0, shift) * f + o
        return fix

