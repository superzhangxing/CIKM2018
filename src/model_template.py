import tensorflow as tf
from configs import cfg
from src.record_log import _logger
from abc import abstractmethod
import numpy as np

class ModelTemplate(object):
    def __init__(self,emb_mat,vocab_len,max_sent_len,scope):
        self.scope = scope
        self.global_step = tf.get_variable(name = 'global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0),trainable=False)
        self.emb_mat = emb_mat

        # ---- place holder ----
        self.sent1_token = tf.placeholder(tf.int32,[None,None],name='sent1_token') # batch_size,sent_length
        self.sent2_token = tf.placeholder(tf.int32,[None,None],name='sent2_token') # batch_size,sent_length

        self.gold_label = tf.placeholder(tf.int32,[None],name='gold_label')
        self.is_train = tf.placeholder(tf.bool,[],name='is_train')

        # ---- parameters ----
        self.vocab_len = vocab_len
        self.max_sent_len = max_sent_len
        self.word_embedding_len = cfg.word_embedding_length
        self.hn = cfg.hidden_units_number

        self.output_class = 2 # 0 for different, 1 for same
        self.batch_size = tf.shape(self.sent1_token)[0]
        self.sent1_len = tf.shape(self.sent1_token)[1]   # include pad
        self.sent2_len = tf.shape(self.sent2_token)[1]

        # ---- others ----
        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)
        self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask,tf.int32),axis=1) # token len, exclude pad
        self.sent2_token_len = tf.reduce_sum(tf.cast(self.sent2_token_mask,tf.int32),axis=1)

        self.tensor_dict = {}

        # ---- dynamic learning rate ----   TODO
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_value = cfg.learning_rate # init learning rate

        # ---- start ----
        self.logits = None
        self.loss = None
        self.accuracy = None  # evaluation need to be changed
        self.summary = None
        self.opt = None
        self.train_opt = None

    @abstractmethod
    def build_network(self):
        pass

    def build_loss(self):
        # weight_decay
        with tf.name_scope('weight_decay'):
            for var in set(tf.get_collection('reg_vars', self.scope)):  # store reg vars
                weight_decay = tf.multiply(tf.nn.l2_loss(var),cfg.weight_decay,
                                           name='{}-weight_decay'.format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses',weight_decay) # store losses
        reg_vars = tf.get_collection('losses',self.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.scope)
        _logger.add('regularization var num: %d' % len(reg_vars))
        _logger.add('trainable var num: %d' % len(trainable_vars))
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.stop_gradient(self.gold_label),
            logits=self.logits
        )
        tf.add_to_collection('losses',tf.reduce_mean(losses,name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses',self.scope),name='loss')
        tf.summary.scalar(loss.op.name,loss)
        tf.add_to_collection('ema/scalar',loss)

        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits,-1), tf.int32),
            self.gold_label
        )

        return tf.cast(correct,tf.float32)

    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

        # -------------------------- ema --------------------------TODO
        # if True:
        #     self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
        #     self.build_var_ema()
        #
        # if cfg.mode == 'train':
        #     self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
        #     self.build_ema()
        self.summary = tf.summary.merge_all()

        # -------------------------- optimization ----------------
        if cfg.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\' ' % cfg.optimizer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # trainable param num:
        # print params num:
        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'): continue   # emb_mat para excluded here
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        _logger.add('Trainable parameters number: %d: '%all_params_num)

        self.train_op = self.opt.minimize(self.loss, self.global_step, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.scope))

    def build_var_ema(self):
        """TODO"""
        ema_op = self.var_ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        """TODO"""
        return

    def step(self, sess, batch_samples, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')

        cfg.time_counter.add_start()
        if get_summary:
            loss, summary, train_op = sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        cfg.time_counter.add_stop()

        return loss, summary, train_op

    # update learning rate
    def update_learning_rate(self,current_dev_loss, global_step, lr_decay_factor=0.7):
        if cfg.dy_lr:
            method = 1
            if method == 0:
                # TODO
                return
            elif method == 1:
                if self.learning_rate_value <5e-6:
                    return
                if global_step % 5000 == 0:
                    self.learning_rate_value *= lr_decay_factor
        return

    def get_feed_dict(self, sample_batch, data_type='train'):
        # max lens
        sent1_len, sent2_len = 0,0
        for sample in sample_batch:
            sent1_len = max(sent1_len, len(sample['sent1_token_id_es']))
            sent2_len = max(sent2_len, len(sample['sent2_token_id_es']))

        # token , char-level blocked
        sent1_token_b = []  # a batch size sentence token list, b --> batch
        sent2_token_b = []
        for sample in sample_batch:
            sent1_token = np.zeros([sent1_len], cfg.intX)
            for idx_t,token in enumerate(sample['sent1_token_id_es']):
                sent1_token[idx_t] = token
            sent2_token = np.zeros([sent2_len], cfg.intX)
            for idx_t,token in enumerate(sample['sent2_token_id_es']):
                sent2_token[idx_t] = token

            sent1_token_b.append(sent1_token)
            sent2_token_b.append(sent2_token)
        sent1_token_b = np.stack(sent1_token_b)
        sent2_token_b = np.stack(sent2_token_b)

        # label
        gold_label_b = []
        for sample in sample_batch:
            gold_label_int = None
            if sample['label'] == 1:  # entailment
                gold_label_int = 1
            elif sample['label'] == 0: # contradiction
                gold_label_int = 0
            assert  gold_label_int is not None
            gold_label_b.append(gold_label_int)
        gold_label_b = np.stack(gold_label_b).astype(cfg.intX)

        feed_dict = {
            self.sent1_token: sent1_token_b,
            self.sent2_token: sent2_token_b,
            self.gold_label:gold_label_b,
            self.is_train:True if data_type=='train' else False,
            self.learning_rate : self.learning_rate_value
        }

        return feed_dict
