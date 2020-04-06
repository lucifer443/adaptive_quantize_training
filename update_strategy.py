# 实现四种自适应量化的策略：
#       1、fix_all:量化位宽（bitnum）与量化参数更新间隔（interval）均固定
#       2、dynamic_interval:interval自适应，bitnum固定
#       3、dynamic_bitnum:interval固定，bitnum自适应
#       4、dynamic_all:interval与bitnum均自适应
# ==============================================================================
import tensorflow as tf
from quantize_utils import compute_quant_param, float2fix

def fix_all(data, new_shift, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）与量化参数更新间隔（interval）均固定
    """
    update_step = global_step + config["interval"]
    return update_step, bitnum, m


# TODO dynamic_interval
def dynamic_interval(data, new_shift, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）固定，量化参数更新间隔（interval）自适应
    """
    pass


# TODO dynamic_bitnum
def dynamic_bitnum(data, new_shift, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）自适应，量化参数更新间隔（interval）固定
    """
    pass


# TODO dynamic_bitnum
def dynamic_all(data, new_shift, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）与量化参数更新间隔（interval）均自适应
    """
    pass


quantize_strategy = {"fix_all": fix_all}


# def _compute_interval(diff, shift, m, config):
#     new_m = config["alpha"] * shift + (1 - config["alpha"]) * m
#     interval = config["beta"] / diff - config["gamma"]
#     interval = tf.cast(tf.maximum(interval, 1), tf.int64)
#     interval = tf.minimum(interval, config["step_per_epoch"])
#     return interval, new_m


def _compute_interval_method1(data, shift, m, config):
    diff = config["alpha"] * tf.abs(shift - m)

    new_m = config["alpha"] * shift + (1 - config["alpha"]) * m
    interval = config["beta"] / diff - config["gamma"]
    interval = tf.cast(tf.maximum(interval, 1), tf.int64)
    interval = tf.minimum(interval, config["step_per_epoch"])
    return interval, new_m


def _compute_interval_method2(data, shift, m, config):
    diff1 = config["alpha"] * tf.abs(shift - m)
    quant_data = float2fix(data, shift, 1, 0)
    metrics = _compute_mean_diff(data, quant_data)
    diff2 = config["delta"] * metrics**2
    diff = tf.maximum(diff1, diff2)

    new_m = config["alpha"] * shift + (1 - config["alpha"]) * m
    interval = config["beta"] / diff - config["gamma"]
    interval = tf.cast(tf.maximum(interval, 1), tf.int64)
    interval = tf.minimum(interval, config["step_per_epoch"])
    return interval, new_m


def _compute_mean_diff(raw, quant):
    raw_mean = tf.reduce_mean(tf.abs(raw))
    quant_mean = tf.reduce_mean(tf.abs(quant))
    mean_diff = tf.log1p(tf.abs(tf.divide(raw_mean - quant_mean, raw_mean))) / tf.log(2.0)
    return mean_diff


def _compute_bitnum(data, ):












