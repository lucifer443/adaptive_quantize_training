import tensorflow as tf
from quantize_utils import compute_quant_param, float2fix


def create_quantize_graph(config, graph=tf.get_default_graph()):
    """
    全图量化
    :param config: 量化方式配置
    :param graph: 需要量化的graph
    :return: 添加量化算子的graph
    """
    ops = graph.get_operations()
    quant_tensors = {}  # 存放已经量化过的tensor，避免重复量化
    quantize_func = quantize_for_train if config["is_training"] else quantize_for_eval

    # 遍历所有op,对前向算子进行量化
    for op in ops:
        if "gradient" not in op.name and op.type in config["forward_quant_ops"]:
            inp1, inp2 = op.inputs._inputs
            quant_inp1 = quantize_func(inp1, config) if inp1.name not in quant_tensors else quant_tensors[inp1.name]
            quant_inp2 = quantize_func(inp2, config) if inp2.name not in quant_tensors else quant_tensors[inp2.name]
            quant_tensors[inp1.name] = quant_inp1
            quant_tensors[inp2.name] = quant_inp2
            tf.contrib.graph_editor.reroute_ts([quant_inp1, quant_inp2], op.inputs._inputs, can_modify=op)

    # 遍历所有op,对所有反向算子进行量化
    for op in ops:
        if "gradient" not in op.name and op.type in config["backward_quant_ops"]:
            new_inputs = []
            for inp in op.inputs._inputs:
                if "ShapeN" not in inp.name:
                    quant_inp = quantize_func(inp, config) if inp.name not in quant_tensors else quant_tensors[inp.name]
                    quant_tensors[inp.name] = quant_inp
                    new_inputs.append(quant_inp)
                else:
                    new_inputs.append(inp)
            tf.contrib.graph_editor.reroute_ts(new_inputs, op.inputs._inputs, can_modify=op)

    return graph


def quantize_for_train(data, config):
    """
    对数据进行量化处理，返回量化处理后的数据
    :param data: 需要量化处理的数据
    :param config: 量化方式配置（dict）
    :return: 量化处理后的数据
    """
    shift, scale, offset, bitnum, update_step, m = _init_quant_param(data.name[:-2], bitnum=config["bitnum"])
    global_step = tf.train.get_or_create_global_step()
    new_shift, new_scale, new_offset, \
    new_bitnum, new_update_step, new_m = tf.cond(tf.equal(global_step, update_step) | tf.less(global_step, 2),
                                                 lambda: get_new_quantize_param(data, global_step, bitnum, m, config),
                                                 lambda: shift, scale, offset, bitnum, update_step, m)

    # 更新量化参数
    assign_shift = tf.assign(shift, new_shift)
    assign_scale = tf.assign(scale, new_scale)
    assign_offset = tf.assign(offset, new_offset)
    assign_bitnum = tf.assign(bitnum, new_bitnum)
    assign_update_step = tf.assign(update_step, new_update_step)
    assign_m = tf.assign(m, new_m)
    with tf.control_dependencies([assign_shift, assign_scale, assign_offset,
                                  assign_bitnum, assign_update_step, assign_m]):
        quantize_data = float2fix(data, shift, scale, offset)
    return quantize_data


def quantize_for_eval(data, config):
    shift, scale, offset = compute_quant_param(data, bitnum=config["bitnum"], ifscale=config["ifscale"],
                                       ifoffset=config["ifoffset"], ifchannel=config["ifchannel"])
    quantize_data = float2fix(data, shift, scale, offset)
    return quantize_data


def _init_quant_param(suffix, bitnum=8):
    """
    生成用于存放量化参数的变量
    :param suffix: 量化参数名称前缀
    :return: 量化参数所存变量
    """
    shift = tf.Variable(0, trainable=False, name="%s_shift" % suffix, dtype=tf.float32)
    scale = tf.Variable(1, trainable=False, name="%s_scale" % suffix, dtype=tf.float32)
    offset = tf.Variable(0, trainable=False, name="%s_offset" % suffix, dtype=tf.float32)
    bitnum = tf.Variable(bitnum, trainable=False, name="%s_offset" % suffix, dtype=tf.int8)
    update_step = tf.Variable(1, trainable=False, name="%s_offset" % suffix, dtype=tf.int64)
    m = tf.Variable(100, trainable=False, name="%s_m" % suffix, dtype=tf.float32)
    tf.add_to_collection("quantize_shift", shift)
    tf.add_to_collection("quantize_scale", scale)
    tf.add_to_collection("quantize_offset", offset)
    tf.add_to_collection("quantize_bitnum", bitnum)
    tf.add_to_collection("quantize_update_step", update_step)
    tf.add_to_collection("quantize_m", m)
    return shift, scale, offset, bitnum, update_step, m

