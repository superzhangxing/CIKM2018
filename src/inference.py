from configs import cfg
from src.record_log import _logger
import numpy as np
import tensorflow as tf

class Inference(object):
    def __init__(self, model):
        self.model = model

    def get_inference(self, sess, dataset_obj):
        _logger.add()
        _logger.add('getting inference result for %s' % dataset_obj.data_type)

        logits_list = []
        prob_list = []
        for sample_batch, _, _, _ in dataset_obj.generate_batch_sample_iter():
            feed_dict = self.model.get_feed_dict(sample_batch, 'infer')
            logits= sess.run(self.model.logits, feed_dict=feed_dict)
            logits_list.append(np.argmax(logits, -1))
            prob_list.append(np.exp(logits[:,1])/(np.exp(logits[:,0]) + np.exp(logits[:,1])))

        logits_array = np.concatenate(logits_list, 0)
        prob_array = np.concatenate(prob_list, 0)

        return logits_array, prob_array

    def save_inference(self, prob_array, file_path = cfg.infer_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for i in range(prob_array.size):
                f.write(str(prob_array[i]))
                f.write('\n')


