import argparse
import os
from os.path import join
from src.time_counter import TimeCounter


class Configs(object):
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = join(self.project_dir,'dataset')

        # ------------ parsing input arguments --------------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ("yes", "true", "t", "1")))

        # @----------- control ------------------------
        parser.add_argument('--network_type', type=str, default='disan', help='network type')
        parser.add_argument('--log_period', type=int, default=2000, help='save tf summary period')
        parser.add_argument('--save_period', type=int, default=3000, help='')
        parser.add_argument('--eval_period', type=int, default=500, help='evaluation period')
        parser.add_argument('--load_model', type='bool', default=False, help='do not use')
        parser.add_argument('--load_step', type=int, default=None, help='do not use')
        parser.add_argument('--load_path', type=str, default=None, help='specify which pre-trained model to be load')

        # @----------- training ----------------------------
        parser.add_argument('--max_epoch', type=int, default=100, help='max epoch number')
        parser.add_argument('--num_steps', type=int, default=40000, help='max steps number')
        parser.add_argument('--train_batch_size', type=int, default=32, help='train batch size')
        parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')
        parser.add_argument('--optimizer', type=str, default='adadelta', help='chose an optimizer[adam|adadelta]')
        parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
        parser.add_argument('--dy_lr', type='bool', default=False, help='if decay lr during training')
        parser.add_argument('--lr_decay', type=float, default=False, help='learning rate decay')
        parser.add_argument('--dropout', type=float, default=1.0, help='dropout keep prob')
        parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay factor/l2 decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='learning rate') # ema
        parser.add_argument('--decay', type=float, default=0.9, help='summary decay') # ema

        # @----------- text processing ----------------------
        parser.add_argument('--word_embedding_length', type=int, default=300, help='word embedding length')
        parser.add_argument('--lower_word', type='bool', default=True, help='')
        parser.add_argument('--sent_len_rate', type=float, default=0.97, help='delete too long sentences') # need to statistic sentences length information both in train dataset and test dataset

        # @----------- network ------------------------------
        parser.add_argument('--hidden_units_number', type=int, default=300, help='hidden units number')

        parser.set_defaults(shuffer=True)
        self.args = parser.parse_args()

        ##  ---------- to member variables ------------------
        for key,value in self.args.__dict__.items():
            if key not in ['test','shuffer']:
                exec('self.%s = self.args.%s' % (key,key))

        # ------------ name ---------------------------------
        self.train_data_name = 'cikm_english_train_20180516.txt'
        self.dev_data_name = 'cikm_spanish_train_20180516.txt'
        self.test_data_name = 'cikm_test_a_20180516.txt'
        self.unlabeled_data_name = 'cikm_unlabel_spanish_train_20180516.txt'
        self.emb_es_name = 'wiki.es.vec'
        self.emb_en_name = 'wiki.en.vec'

        self.processed_name = 'processed' + '.pickle'

        self.dict_name = 'dicts'

        self.model_name = self.network_type
        self.model_ckpt_name = 'modelfile.ckpt'

        # ------------ dir ----------------------------------
        self.result_dir = self.mkdir(self.project_dir, 'result')
        self.dict_dir = self.mkdir(self.result_dir, 'dict')
        self.processed_dir = self.mkdir(self.result_dir, 'processed_data')

        self.all_model_dir = self.mkdir(self.result_dir, 'model')
        self.model_dir = self.mkdir(self.all_model_dir, self.model_name)
        self.log_dir = self.mkdir(self.all_model_dir, 'log')
        self.summary_dir = self.mkdir(self.all_model_dir, 'summary')
        self.ckpt_dir = self.mkdir(self.model_dir, 'ckpt')



        # ------------ path ---------------------------------
        self.train_data_path = join(self.dataset_dir,self.train_data_name)
        self.dev_data_path = join(self.dataset_dir, self.dev_data_name)
        self.test_data_path = join(self.dataset_dir, self.test_data_name)
        self.unlabeled_data_path = join(self.dataset_dir, self.unlabeled_data_name)
        self.emb_en_path = join(self.dataset_dir, self.emb_en_name)
        self.emb_es_path = join(self.dataset_dir, self.emb_es_name)


        self.processed_path = join(self.processed_dir, self.processed_name)
        self.dict_path = join(self.dict_dir, self.dict_name)
        self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)

        # ------------ other --------------------------------
        self.floatX = 'float32'
        self.intX = 'int32'
        self.time_counter = TimeCounter()


    def get_params_str(self, params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '_' + str(eval('self.args.' + paramsStr))
        return model_params_str

    def mkdir(self, *args):
        dirPath = join(*args)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        return dirPath

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        fileName = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return fileName

cfg = Configs()