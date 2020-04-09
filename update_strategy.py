# 实现四种自适应量化的策略：
#       1、fix_all:量化位宽（bitnum）与量化参数更新间隔（interval）均固定
#       2、dynamic_interval:interval自适应，bitnum固定
#       3、dynamic_bitnum:interval固定，bitnum自适应
#       4、dynamic_all:interval与bitnum均自适应
# ==============================================================================
import tensorflow as tf
from quantize_utils import compute_quant_param, float2fix

def fix_all(data, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）与量化参数更新间隔（interval）均固定
    """
    new_shift, new_f, new_o = compute_quant_param(data, bitnum)
    update_step = global_step + config["interval"]
    return new_shift, new_f, new_o, update_step, bitnum, m


def dynamic_interval(data, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）固定，量化参数更新间隔（interval）自适应
    """
    new_shift, new_f, new_o = compute_quant_param(data, bitnum)
    new_m = config["alpha"] * new_shift + (1 - config["alpha"]) * m
    update_step = global_step + tf.cond(
        tf.less(global_step, int(config["step_per_epoch"]/100)),
        lambda: tf.constant(1, dtype=tf.int64),
        lambda: tf.cond(
            tf.less(global_step, int(config["step_per_epoch"])),
            lambda: _compute_interval(data, new_shift, new_m, config),
            lambda: tf.constant(config["steps_per_epoch"], dtype=tf.int64)
        )
    )
    return new_shift, new_f, new_o, update_step, bitnum, new_m


def dynamic_bitnum(data, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）自适应，量化参数更新间隔（interval）固定
    """
    new_bitnum = _compute_bitnum(data, bitnum, config)
    new_shift, new_f, new_o = compute_quant_param(data, new_bitnum)
    new_m = config["alpha"] * new_shift + (1 - config["alpha"]) * (m + tf.cast(bitnum - new_bitnum, tf.float32))
    update_step = global_step + config["interval"]
    return new_shift, new_f, new_o, update_step, new_bitnum, new_m


def dynamic_all(data, global_step, bitnum, m, config):
    """
    量化位宽（bitnum）与量化参数更新间隔（interval）均自适应
    """
    new_bitnum = _compute_bitnum(data, bitnum, config)
    new_shift, new_f, new_o = compute_quant_param(data, new_bitnum)
    new_m = config["alpha"] * new_shift + (1 - config["alpha"]) * (m + tf.cast(bitnum - new_bitnum, tf.float32))
    update_step = global_step + tf.cond(
        tf.less(global_step, int(config["step_per_epoch"]/100)),
        lambda: tf.constant(1, dtype=tf.int64),
        lambda: tf.cond(
            tf.less(global_step, int(config["step_per_epoch"])),
            lambda: _compute_interval(data, new_shift, new_m, config),
            lambda: tf.constant(config["steps_per_epoch"], dtype=tf.int64)
        )
    )
    return new_shift, new_f, new_o, update_step, new_bitnum, new_m


quantize_strategy = {"fix_all": fix_all,
                     "dynamic_interval": dynamic_interval,
                     "dynamic_bitnum": dynamic_bitnum,
                     "dynamic_all": dynamic_all}


def _compute_interval(data, shift, m, config):
    """
    计算量化间隔
    """
    diff1 = config["alpha"] * tf.abs(shift - m)
    quant_data = float2fix(data, shift, 1, 0)
    metrics = _compute_mean_diff(data, quant_data)
    diff2 = config["delta"] * metrics**2
    diff = tf.maximum(diff1, diff2)

    interval = config["beta"] / diff - config["gamma"]
    interval = tf.cast(tf.maximum(interval, 1), tf.int64)
    interval = tf.minimum(interval, config["step_per_epoch"])
    return interval


def _compute_mean_diff(raw, quant):
    """
    衡量量化前后数据分布差异
    """
    raw_mean = tf.reduce_mean(tf.abs(raw))
    quant_mean = tf.reduce_mean(tf.abs(quant))
    mean_diff = tf.log1p(tf.abs(tf.divide(raw_mean - quant_mean, raw_mean))) / tf.log(2.0)
    return mean_diff


def _compute_bitnum(data, bitnum, config):
    """
    计算新的量化位宽
    """
    shift, f, o = compute_quant_param(data, bitnum)
    outdata = float2fix(data, shift, f, o, bitnum=bitnum)
    diff = _compute_mean_diff(data, outdata)

    loop = [diff, bitnum, data]
    cond = lambda diff, bitnum, data: tf.greater(diff, config["ths"]) & tf.less(bitnum, 33)
    body = lambda diff, bitnum, data: _loop_body(diff, bitnum, data)
    diff, bitnum, data = tf.while_loop(cond, body, loop)
    return bitnum

def _loop_body(diff, bitnum, data):
    bitnum = bitnum + 8
    new_shift, f, o = compute_quant_param(data, bitnum)
    outdata = float2fix(data, new_shift, f, o, bitnum=bitnum)
    diff = _compute_mean_diff(data, outdata)
    return diff, bitnum, data













