import tensorflow as tf
from functools import reduce
from operator import mul


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

# ---------------------------- DiSAN interface ----------------------------

def disan(rep_tensor,rep_mask,scope=None,
          keep_prob = 1.0, is_train=None, weight_decay=0.,
          activation='elu', tensor_dict={},name=None):
    with tf.variable_scope(scope or 'DiSAN'):
        with tf.variable_scope('ct_attn'):
            fw_res = directional_attention_with_dense(rep_tensor, rep_mask,'forward',
                                                      'dir_attn_fw', keep_prob, is_train,
                                                      weight_decay, activation, tensor_dict, name+'_fw_attn')
            bw_res = directional_attention_with_dense(rep_tensor, rep_mask, 'backward',
                                                      'dir_attn_bw', keep_prob, is_train,
                                                      weight_decay, activation, tensor_dict, name+'_bw_attn')
            seq_rep = tf.concat([fw_res, bw_res], -1)  # batch_size, sent_len, 2*word_embedding_len

        # multi dimensional source2token attention
        with tf.variable_scope('sent_enc_attn'):
            sent_rep = multi_dimensional_attention(seq_rep,rep_mask,'multi_dimensional_attention',
                                                   keep_prob, is_train, weight_decay, activation, tensor_dict,
                                                   name = name+'_attn')

            return sent_rep

def directional_attention_with_dense(rep_tensor, rep_mask,direction=None, scope=None,
                                     keep_prob=1., is_train=None, weight_decay=0.,
                                     activation = 'elu', tensor_dict = None, name=None):
    def scaled_tanh(x,scale=5.):
        return scale*tf.nn.tanh(1./scale*x)

    batch_size, sent_len, word_embedding_len = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2] # a different size is alternative
    with tf.variable_scope(scope or 'directional_attention_%' % 'direction' or 'diag'):
        # mask generation
        sent_len_indices = tf.range(sent_len, dtype=tf.int32)
        sent_len_col, sent_len_row = tf.meshgrid(sent_len_indices,sent_len_indices)
        if direction is None:  #  True-->0, False-->-inf
            direct_mask = tf.cast(tf.diag(-tf.ones([sent_len], tf.int32))+1,tf.bool) # sent_len,sent_len
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sent_len_row, sent_len_col)
            else:
                direct_mask = tf.greater(sent_len_col, sent_len_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask,0),[batch_size,1,1]) # batch_size, sent_len, sent_len
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask,1),[1,sent_len,1]) # batch_size, sent_len, sent_len
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile) # batch_size, sent_len, sent_len

        # non-linear  ,x-->h
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation, False, weight_decay,
                                 keep_prob, is_train) # batch_size, sent_len, ivec
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1),[1, sent_len, 1, 1])  # batch_size,sent_len,sent_len,ivec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):
            f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent') # batch_size, sent_len, ivec
            dependent_etd = tf.expand_dims(dependent, axis=1) # batch_size, 1, sent_len, ivec
            head = linear(rep_map_dp, ivec, False, 'linear_head')
            head_etd = tf.expand_dims(head, axis=2) # batch_size, sent_len, 1, ivec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)

            logits_mask = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_mask, 2)
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)

        # output, fusion
        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, weight_decay, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, weight_decay, keep_prob, is_train) +
                o_bias
            )
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attention
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate

        return output


# multi dimensional source2token attention
def multi_dimensional_attention(rep_tensor, rep_mask, scope=None, keep_prob=1.,
                                is_train=None, weight_decay=0., activation='elu',
                                tensor_dict=None, name=None):
    batch_size, sent_len, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]

    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, weight_decay, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', activation,
                              False, weight_decay, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked,1) # bs, sl, vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1) # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


# bn-->batch norm
def bn_dense_layer(input_tensor,hn,bias,bias_start=0.,scope=None,
                   activation = 'relu', enable_bn=True,
                   weight_decay = 0., keep_prob = 1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map', False,
                            weight_decay, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(linear_map, center=True, scale=True,
                                                      is_training = is_train, scope='bn')
        return activation_func(linear_map)

# dropout
def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or 'dropout'):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x,keep_prob,noise_shape=noise_shape,seed=seed)
            out = tf.cond(is_train, lambda:d, lambda :a)
            return out
        return x

# linear
def linear(args, output_size, is_bias, bias_start=0., scope=None, squeeze=False,
           weight_decay = 0., input_keep_prob=1.0, is_train=None):
    if args is None or (isinstance(args,(tuple,list)) and not args):
        raise ValueError("'args' must be specified")

    if not isinstance(args, (tuple,list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # flat_args -->list, flatten(arg,1)-->tensor,rank=2
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda:tf.nn.dropout(arg,input_keep_prob), lambda: arg)
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, is_bias, bias_start, scope)# flat_out--> tensor with rank 2,[bs*st,de]
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if weight_decay: # reg weight para
        add_reg_without_bias()

    return out



def _linear(xs, output_size, is_bias, bias_start=0., scope=None):
    """
    :param xs: rank is 3
    :param output_size:
    :param is_bias:
    :param bias_start:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)   # xs --> list, x --> tensor
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size], dtype=tf.float32)

        if is_bias:
            bias = tf.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x,W) + bias
        else:
            out = tf.matmul(x,W)
        return out

# flat tensor, last 'keep' ranks is retained, first serval ranks should be exec mul op
#  tensor-->[2,3,4,5]; keep = 2, out-->[6,4,5];keep=1,out-->[24,5]
def flatten(tensor, keep):
    fixed_shaped = tensor.get_shape().as_list()
    start = len(fixed_shaped) - keep
    left = reduce(mul, [fixed_shaped[i] or tf.shape(tensor)[i] for i in range(start)]) #
    out_shape = [left] + [fixed_shaped[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shaped))]
    flat = tf.reshape(tensor, out_shape)
    return flat

# reconstruct tensor
# out-->rank_ref-keep+dim_reduced_keep
# 'keep' means last 'keep' ranks are retained
def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # orginal shape
    tensor_shape = tensor.get_shape().as_list() #current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name = name or 'mask_for_high_rank')

def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32))*VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')

#
def selu(x):
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

#  add vars to reg collections, bias excluded
def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter