import math
import tensorflow as tf
import os

from configs import cfg
from src.dataset import Dataset
from src.file import load_file, save_file
from src.record_log import _logger
from src.graph_handler import GraphHandler
from src.evaluator import Evaluator
from src.perform_recorder import PerformRecorder
from src.inference import Inference
from src.util.preprocess_data import PreprocessData

# choose model
network_type = cfg.network_type
if network_type == 'disan':
    from src.model_disan import ModelDiSAN as Model
elif network_type == 'rnn_bi_lstm':
    from src.model_bi_lstm import ModelBiLSTM as Model

model_type_set = ['disan','rnn_bi_lstm']

def train():
    # output_model_params()

    # need to fixed, reusability of data
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        train_data_obj = Dataset(cfg.train_data_path, 'train', language_type='es',
                         unlabeled_file_path=cfg.unlabeled_data_path, emb_file_path=cfg.emb_es_path)
        dev_data_obj = Dataset(cfg.dev_data_path, 'dev', dicts=train_data_obj.dicts)
        test_data_obj = Dataset(cfg.test_data_path, 'test', dicts=train_data_obj.dicts)

        save_file({'train_data_obj':train_data_obj, 'dev_data_obj':dev_data_obj, 'test_data_obj':test_data_obj},
                  cfg.processed_path)

        train_data_obj.save_dict(cfg.dict_path)
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    emb_mat_token = train_data_obj.emb_mat_token

    with tf.variable_scope(network_type) as scope:
        if network_type in model_type_set:
            model = Model(emb_mat_token, len(train_data_obj.dicts['es']), 100, scope.name)

    # TODO
    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)
    performRecorder = PerformRecorder(3)

    graph_config = tf.ConfigProto()
    sess = tf.Session(config=graph_config)
    graphHandler.initialize(sess)


    # begin training
    steps_per_epoch = int(math.ceil(1.0 * train_data_obj.sample_num / cfg.train_batch_size))
    num_steps = cfg.num_steps or steps_per_epoch*cfg.max_epoch

    global_step = 0

    for sample_batch, batch_num, data_round, idx_b in train_data_obj.generate_batch_sample_iter(num_steps):
        global_step = sess.run(model.global_step) + 1
        if_get_summary = global_step % (cfg.log_period or steps_per_epoch) == 0
        loss, summary, train_op = model.step(sess, sample_batch, get_summary=if_get_summary)
        if global_step % 10 ==0 or global_step==1:
            _logger.add('data round: %d: %d/%d, global_step:%d -- loss:%.4f' %
                        (data_round, idx_b, batch_num, global_step, loss))

        if if_get_summary:
            graphHandler.add_summary(summary, global_step)

        # Occasional evaluation
        #if global_step > int(cfg.num_steps - 100000) and (global_step % (cfg.eval_period or steps_per_epoch) == 0):
        if True:  # debug
            # ---- dev ----
            dev_loss, dev_accu =evaluator.get_evaluation(sess, dev_data_obj, global_step)
            _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' % (dev_loss, dev_accu))

            # ---- test ----
            test_loss, test_accu = evaluator.get_evaluation(
                sess, test_data_obj, global_step
            )
            _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' % (test_loss, test_accu))

            model.update_learning_rate(dev_loss, cfg.lr_decay)
            is_in_top, deleted_step = performRecorder.update_top_list(global_step, dev_accu, sess)

        this_epoch_time, mean_epoch_time = cfg.time_counter.update_data_round(data_round)
        if this_epoch_time is not None and mean_epoch_time is not None:
            _logger.add('##> this epoch time: %f, mean epoch time: %f' % (this_epoch_time, mean_epoch_time))

    # TODO
    # do_analyse

# deprecated
def test():

    assert cfg.load_path is not None

    #TODO
    loadFile = False
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        train_data_obj = Dataset(cfg.train_data_path, 'train', language_type='es',
                         unlabeled_file_path=cfg.unlabeled_data_path, emb_file_path=cfg.emb_es_path)
        dev_data_obj = Dataset(cfg.dev_data_path, 'dev', dicts=train_data_obj.dicts)
        test_data_obj = Dataset(cfg.test_data_path, 'test', dicts=train_data_obj.dicts)

        save_file({'train_data_obj':train_data_obj, 'dev_data_obj':dev_data_obj, 'test_data_obj':test_data_obj},
                  cfg.processed_path)

        train_data_obj.save_dict(cfg.dict_path)
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    emb_mat_token = train_data_obj.emb_mat_token

    with tf.variable_scope(network_type) as scope:
        if network_type in model_type_set:
            model = Model(emb_mat_token, len(train_data_obj.dicts['es']), 100, scope.name)

    # TODO
    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)

    graph_config = tf.ConfigProto()
    sess = tf.Session(config=graph_config)
    graphHandler.initialize(sess)

    # ---- dev ----
    dev_loss, dev_accu = evaluator.get_evaluation(sess, dev_data_obj, None)
    _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' % (dev_loss, dev_accu))

    # ---- test ----
    test_loss, test_accu = evaluator.get_evaluation(sess, test_data_obj, None)
    _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' % (test_loss, test_accu))

    # ---- train ----
    train_loss, train_accu = evaluator.get_evaluation(sess, train_data_obj, None)
    _logger.add('--> for train, loss: %.4f, accuracy: %.4f' % (train_loss, train_accu))

def infer():
    # load infer data, need to fix
    #TODO
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        train_data_obj = Dataset(cfg.train_data_path, 'train', language_type='es',
                         unlabeled_file_path=cfg.unlabeled_data_path, emb_file_path=cfg.emb_es_path)
        dev_data_obj = Dataset(cfg.dev_data_path, 'dev', dicts=train_data_obj.dicts)
        test_data_obj = Dataset(cfg.test_data_path, 'test', dicts=train_data_obj.dicts)

        save_file({'train_data_obj':train_data_obj, 'dev_data_obj':dev_data_obj, 'test_data_obj':test_data_obj},
                  cfg.processed_path)

        train_data_obj.save_dict(cfg.dict_path)
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    infer_data_obj = Dataset(cfg.infer_data_path, 'infer', dicts=train_data_obj.dicts)

    # load model
    emb_mat_token = train_data_obj.emb_mat_token   # need to restore model

    with tf.variable_scope(network_type) as scope:
        if network_type in model_type_set:
            model = Model(emb_mat_token, len(train_data_obj.dicts['es']), 100, scope.name)

    graphHandler = GraphHandler(model)
    #evaluator = Evaluator(model)
    inference = Inference(model)

    graph_config = tf.ConfigProto()
    sess = tf.Session(config=graph_config)
    graphHandler.initialize(sess)

    saver = tf.train.Saver()
    step = 6500
    model_path = os.path.join(cfg.ckpt_dir, 'top_result_saver_step_%d.ckpt' % step)
    saver.restore(sess, model_path)
    logits_array, prob_array = inference.get_inference(sess, test_data_obj)

    inference.save_inference(prob_array, cfg.infer_result_path)

def preprocess():
    preprocess_data = PreprocessData(cfg.org_train_data_path)
    preprocess_data.downsampling(10)

def main(_):
    if cfg.mode == 'train':
        train()
    elif cfg.mode == 'infer':
        infer()
    elif cfg.mode == 'preprocess':
        preprocess()
    else:
        raise RuntimeError('no running mode named as %s'% cfg.mode)

if __name__ == '__main__':
    tf.app.run()