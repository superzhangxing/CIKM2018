import tensorflow as tf
import numpy as np

from src.model_template import ModelTemplate
from src.func import disan, linear
from configs import cfg
from src.record_log import  _logger

class ModelDiSAN(ModelTemplate):
    def __init__(self,emb_mat, vocab_len, max_sent_len, scope):
        super(ModelDiSAN,self).__init__(emb_mat,vocab_len,max_sent_len,scope)
        self.update_tensor_add_ema_and_opt()


    def build_network(self):
        _logger.add()
        _logger.add('building %s nerual network structure ...' % cfg.network_type)

        vocab_len = self.vocab_len
        max_sent_len = self.max_sent_len
        word_embedding_len = self.word_embedding_len
        hn = self.hn

        batch_size = self.batch_size
        sent1_len = self.sent1_len #include pad
        sent2_len = self.sent2_len

        # --------------------- embedding mat ------------------------------------
        with tf.variable_scope('emb'):
            # token_emb_mat, 0-->empty, 1-->unknown token
            token_emb_mat = tf.get_variable('token_emb_mat',initializer=tf.constant(self.emb_mat,dtype=tf.float32),dtype=tf.float32,trainable=True)

            sent1_emb = tf.nn.embedding_lookup(token_emb_mat, self.sent1_token) # batch_size,sent1_len,word_embedding_len
            sent2_emb = tf.nn.embedding_lookup(token_emb_mat, self.sent2_token)

            self.tensor_dict['sent1_emb'] = sent1_emb
            self.tensor_dict['sent2_emb'] = sent2_emb

        # -------------------- sentence encoding ---------------------------------
        with tf.variable_scope('sent_encoding'):
            sent1_rep = disan(sent1_emb, self.sent1_token_mask, 'DiSAN', cfg.dropout, self.is_train,
                              cfg.weight_decay, 'elu', self.tensor_dict, 'sent1')
            self.tensor_dict['sent1_rep'] = sent1_rep  # batch_size,2*word_embedding_len

            tf.get_variable_scope().reuse_variables()

            sent2_rep = disan(sent2_emb, self.sent2_token_mask, 'DiSAN', cfg.dropout, self.is_train,
                              cfg.weight_decay, 'elu', self.tensor_dict, 'sent2')
            self.tensor_dict['sent2_rep'] = sent2_rep

        # -------------------- output ---------------------------------------------
        with tf.variable_scope('output'):
            # batch_size, sent_len, 4*2*word_embedding_len
            out_rep = tf.concat([sent1_rep, sent2_rep, sent1_rep-sent2_rep, sent1_rep*sent2_rep], axis=-1)
            pre_output = tf.nn.elu(linear([out_rep], hn, True, 0., scope='pre_output', squeeze=False,
                                          weight_decay=cfg.weight_decay, input_keep_prob=cfg.dropout, is_train=self.is_train))
            logits = linear([pre_output], self.output_class, True, 0., scope='logits', squeeze=False,
                            weight_decay=cfg.weight_decay, input_keep_prob=cfg.dropout, is_train=self.is_train)
            self.tensor_dict['logits'] = logits

        return logits





