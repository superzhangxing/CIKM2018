from configs import cfg
from src.record_log import _logger
import numpy as np
import tensorflow as tf

class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.global_step = model.global_step

        # ---- summary ----
        self.build_summary()
        self.writer = tf.summary.FileWriter(cfg.summary_dir)

    def get_evaluation(self, sess, dataset_obj, global_step=None):
        _logger.add()
        _logger.add('getting evaluation result for %s' % dataset_obj.data_type)

        logits_list, loss_list, accu_list, accu_0_list, accu_1_list, gold_label_list = [], [], [], [], [], []

        for sample_batch, _, _, _ in dataset_obj.generate_batch_sample_iter():
            feed_dict = self.model.get_feed_dict(sample_batch, 'dev')
            logits, loss, accu, accu_0, accu_1, gold_label = sess.run([self.model.logits,self.model.loss, self.model.accuracy,
                                                                       self.model.accuracy_0, self.model.accuracy_1, self.model.gold_label], feed_dict=feed_dict)
            logits_list.append(np.argmax(logits, -1))
            loss_list.append(loss)
            accu_list.append(accu)
            accu_0_list.append(accu_0)
            accu_1_list.append(accu_1)
            gold_label_list.append(gold_label)

        logits_array = np.concatenate(logits_list, 0)
        loss_value = np.mean(loss_list)
        accu_array = np.concatenate(accu_list, 0)
        accu_value = np.mean(accu_array)
        accu_0_array = np.concatenate(accu_0_list, 0)
        accu_0_value = np.mean(accu_0_array)
        accu_1_array = np.concatenate(accu_1_list, 0)
        accu_1_value = np.mean(accu_1_array)
        gold_label_array = np.concatenate(gold_label_list, 0)
        count_gold_label = gold_label_array.size
        count_gold_label_0 = np.count_nonzero(gold_label_array)
        count_gold_label_1 = count_gold_label - count_gold_label_0
        accu_0_value = accu_0_value * count_gold_label / count_gold_label_0
        accu_1_value = accu_1_value * count_gold_label / count_gold_label_1

        if global_step is not None:
            if dataset_obj.data_type == 'train':
                summary_feed_dict = {
                    self.train_loss: loss_value,
                    self.train_accuracy: accu_value,
                    self.train_accuracy_0: accu_0_value,
                    self.train_accuracy_1: accu_1_value
                }
                summary = sess.run(self.train_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            elif dataset_obj.data_type == 'dev':
                summary_feed_dict = {
                    self.dev_loss: loss_value,
                    self.dev_accuracy: accu_value,
                    self.dev_accuracy_0: accu_0_value,
                    self.dev_accuracy_1: accu_1_value
                }
                summary = sess.run(self.dev_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            else:
                summary_feed_dict = {
                    self.test_loss: loss_value,
                    self.test_accuracy: accu_value,
                    self.test_accuracy_0: accu_0_value,
                    self.test_accuracy_1: accu_1_value
                }
                summary = sess.run(self.test_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)

        return loss_value, accu_value, accu_0_value, accu_1_value

    # --- internal use ------
    def build_summary(self):
        with tf.name_scope('train_summaries'):
            self.train_loss = tf.placeholder(tf.float32, [], 'train_loss')
            self.train_accuracy = tf.placeholder(tf.float32, [], 'train_accuracy')
            self.train_accuracy_0 = tf.placeholder(tf.float32, [], 'train_accuracy_0')
            self.train_accuracy_1 = tf.placeholder(tf.float32, [], 'train_accuracy_1')
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss', self.train_loss))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy', self.train_accuracy))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy_0', self.train_accuracy_0))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy_1', self.train_accuracy_1))
            self.train_summaries = tf.summary.merge_all('train_summaries_collection')

        with tf.name_scope('dev_summaries'):
            self.dev_loss = tf.placeholder(tf.float32, [], 'dev_loss')
            self.dev_accuracy = tf.placeholder(tf.float32, [], 'dev_accuracy')
            self.dev_accuracy_0 = tf.placeholder(tf.float32, [], 'dev_accuracy_0')
            self.dev_accuracy_1 = tf.placeholder(tf.float32, [], 'dev_accuracy_1')
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss', self.dev_loss))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy', self.dev_accuracy))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy_0', self.dev_accuracy_0))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy_1', self.dev_accuracy_1))
            self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')

        with tf.name_scope('test_summaries'):
            self.test_loss = tf.placeholder(tf.float32, [], 'test_loss')
            self.test_accuracy = tf.placeholder(tf.float32, [], 'test_accuracy')
            self.test_accuracy_0 = tf.placeholder(tf.float32, [], 'test_accuracy_0')
            self.test_accuracy_1 = tf.placeholder(tf.float32, [], 'test_accuracy_1')
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss', self.test_loss))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_accuracy', self.test_accuracy))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_accuracy_0', self.test_accuracy_0))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_accuracy_1', self.test_accuracy_1))
            self.test_summaries = tf.summary.merge_all('test_summaries_collection')
