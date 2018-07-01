from  configs import cfg
import random
from src.record_log import _logger
from os.path import join
from src.file import save_file
import nltk
import re

PADDING = '<pad>'
UNKNOWN = '<unk>'

class PreprocessData(object):
    def __init__(self, en_trian_file, es_train_file, test_a_file, unlabeled_file, en_vec_file, es_vec_file):
        self.en_train_file = en_trian_file
        self.es_train_file = es_train_file
        self.test_a_file = test_a_file
        self.unlabeled_file = unlabeled_file
        self.en_vec_file = en_vec_file
        self.es_vec_file = es_vec_file

    def build_vocab(self, en_limit=500000, es_limit=500000):
        _logger.add()
        _logger.add('start build vocab ...')
        vocab_es_count = {}
        vocab_en_count = {}
        vocab_es_token2id = {}
        vocab_es_id2token = {}
        vocab_en_token2id = {}
        vocab_en_id2token = {}
        vocab_es_token2id[PADDING] = 0
        vocab_es_token2id[UNKNOWN] = 1
        vocab_es_id2token[0] = PADDING
        vocab_es_id2token[1] = UNKNOWN
        vocab_en_token2id[PADDING] = 0
        vocab_en_token2id[UNKNOWN] = 1
        vocab_en_id2token[0] = PADDING
        vocab_en_id2token[1] = UNKNOWN
        id_es = 2
        id_en = 2

        # generate vocab_count
        _logger.add('build vocab with en train file ...')
        with open(self.en_train_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                sents = line.split('\t')
                sent1_en = sents[0]
                sent1_es = sents[1]
                sent2_en = sents[2]
                sent2_es = sents[3]
                sent1_token_en = tokenize(sent1_en, language='en')
                sent1_token_es = tokenize(sent1_es, language='es')
                sent2_token_en = tokenize(sent2_en, language='en')
                sent2_token_es = tokenize(sent2_es, language='es')
                for token in sent1_token_es+sent2_token_es:
                    token = token.lower() if cfg.lower_word else token
                    vocab_es_count[token] = vocab_es_count.get(token, 0) + 1
                for token in sent1_token_en+sent2_token_en:
                    token = token.lower() if cfg.lower_word else token
                    vocab_en_count[token] = vocab_en_count.get(token, 0) + 1
        _logger.add('build vocab with es train file ...')
        with open(self.es_train_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                sents = line.split('\t')
                sent1_es = sents[0]
                sent1_en = sents[1]
                sent2_es = sents[2]
                sent2_en = sents[3]
                sent1_token_en = tokenize(sent1_en, language='en')
                sent1_token_es = tokenize(sent1_es, language='es')
                sent2_token_en = tokenize(sent2_en, language='en')
                sent2_token_es = tokenize(sent2_es, language='es')
                for token in sent1_token_es+sent2_token_es:
                    token = token.lower() if cfg.lower_word else token
                    vocab_es_count[token] = vocab_es_count.get(token, 0) + 1
                for token in sent1_token_en+sent2_token_en:
                    token = token.lower() if cfg.lower_word else token
                    vocab_en_count[token] = vocab_en_count.get(token, 0) + 1
        _logger.add('build vocab with test a file ...')
        with open(self.test_a_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                sents = line.split('\t')
                sent1_es = sents[0]
                sent2_es = sents[1]
                sent1_token_es = tokenize(sent1_es, language='es')
                sent2_token_es = tokenize(sent2_es, language='es')
                for token in sent1_token_es+sent2_token_es:
                    token = token.lower() if cfg.lower_word else token
                    vocab_es_count[token] = vocab_es_count.get(token, 0) + 1
        _logger.add('build vocab with unlabeled file ...')
        with open(self.unlabeled_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                sents = line.split('\t')
                sent_es = sents[0]
                sent_en = sents[1]
                # sent_token_es = sent_es.split()
                # sent_token_en = sent_en.split()
                sent_token_es = tokenize(sent_es, language='es')
                sent_token_en = tokenize(sent_en, language='en')
                for token in sent_token_es:
                    token = token.lower() if cfg.lower_word else token
                    vocab_es_count[token] = vocab_es_count.get(token, 0) + 1
                for token in sent_token_en:
                    token = token.lower() if cfg.lower_word else token
                    vocab_en_count[token] = vocab_en_count.get(token, 0) + 1

        sorted_vocab_es = sorted(vocab_es_count.items(), key=lambda item: item[1], reverse=True)
        sorted_vocab_en = sorted(vocab_en_count.items(), key=lambda item: item[1], reverse=True)

        #
        for token_pair in sorted_vocab_es:
            vocab_es_id2token[id_es] = token_pair[0]
            vocab_es_token2id[token_pair[0]] = id_es
            id_es += 1

        for token_pair in sorted_vocab_en:
            vocab_en_id2token[id_en] = token_pair[0]
            vocab_en_token2id[token_pair[0]] = id_en
            id_en += 1

        # pre-trained embedding file
        _logger.add('build dict with en vec file ...')
        with open(self.en_vec_file,'r',encoding='utf-8') as f:
            f.readline()  # skip first line
            for line in f:
                if id_en > en_limit:
                    break
                tokens = line.rstrip().split()
                if len(tokens) > 300:
                    token = ' '.join(tokens[:len(tokens) - 300])
                    if cfg.lower_word:
                        token = token.lower()
                    if token not in vocab_en_token2id:
                        vocab_en_id2token[id_en] = token
                        vocab_en_token2id[token] = id_en
                        id_en += 1

        _logger.add('build dict with es vec file ...')
        with open(self.es_vec_file,'r',encoding='utf-8') as f:
            f.readline()  # skip first line
            for line in f:
                if id_es > es_limit:
                    break
                tokens = line.rstrip().split()
                if len(tokens) > 300:
                    token = ' '.join(tokens[:len(tokens) - 300])
                    if cfg.lower_word:
                        token = token.lower()
                    if token not in vocab_es_token2id:
                        vocab_es_id2token[id_es] = token
                        vocab_es_token2id[token] = id_es
                        id_es += 1


        dicts = {}
        dicts['en'] = vocab_en_token2id
        dicts['es'] = vocab_es_token2id
        self.dicts = dicts
        save_file(self.dicts, cfg.dict_path, 'token dict data', 'pickle')
        return vocab_es_count,vocab_es_token2id,vocab_es_id2token,vocab_en_count,vocab_en_token2id,vocab_en_id2token


    # k fold
    def downsampling(self, k):
        _logger.add()
        _logger.add('downsampling ...')
        sample_label = []
        sample_label_0 = []
        sample_label_1 = []
        count_label_0 = 0
        count_label_1 = 0
        with open(self.en_train_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                if len(line) < 2:  # empty line
                    continue
                line_spl = line.split()
                if line[-1] != '\n': # last line
                    line += '\n'
                if int(line_spl[-1]) == 0:
                    count_label_0 += 1
                    sample_label_0.append(line)
                elif int(line_spl[-1]) == 1:
                    count_label_1 += 1
                    sample_label_1.append(line)


        random.shuffle(sample_label_0)
        random.shuffle(sample_label_1)
        if count_label_0<count_label_1:
            sample_label = sample_label + sample_label_0
            sample_label = sample_label + sample_label_1[:count_label_0]
        else:
            sample_label = sample_label + sample_label_0[:count_label_1]
            sample_label = sample_label + sample_label_1

        assert k > 0
        random.shuffle(sample_label)
        len_per_fold = len(sample_label)//k
        sample_k_fold = []
        for i in range(k):
            if i != k-1:
                sample_k_fold.append(sample_label[i*len_per_fold:(i+1)*len_per_fold])
            else:
                sample_k_fold.append(sample_label[i*len_per_fold:])

        # write to file
        train_name = 'train'
        dev_name = 'dev'
        for i in range(k):
            temp_train_name = train_name + '_' + str(i+1)
            temp_dev_name = dev_name + '_' + str(i+1)
            temp_train_path = join(cfg.dataset_dir, temp_train_name)
            temp_dev_path = join(cfg.dataset_dir, temp_dev_name)
            with open(temp_train_path, 'w', encoding='utf-8') as f:
                for j in range(k):
                    if j==i:
                        continue
                    else:
                        f.writelines(sample_k_fold[j])
            with open(temp_dev_path, 'w', encoding='utf-8') as f:
                f.writelines(sample_k_fold[i])

    # upsampling , save as  k fold, dev have repeative data in train
    def upsampling(self, k):
        _logger.add()
        _logger.add('upsampling ...')
        sample_label = []
        sample_label_0 = []
        sample_label_1 = []
        count_label_0 = 0
        count_label_1 = 0
        with open(self.en_train_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                if len(line) < 2:  # empty line
                    continue
                line_spl = line.split()
                if line[-1] != '\n': # last line
                    line += '\n'
                if int(line_spl[-1]) == 0:
                    count_label_0 += 1
                    sample_label_0.append(line)
                elif int(line_spl[-1]) == 1:
                    count_label_1 += 1
                    sample_label_1.append(line)

        random.shuffle(sample_label_0)
        random.shuffle(sample_label_1)

        # upsample
        if count_label_0<count_label_1:
            sample_label = sample_label + sample_label_1
            rate = count_label_1/count_label_0
            while(rate > 1):
                sample_label = sample_label + sample_label_0
                rate -= 1
            if rate > 0:
                sample_label = sample_label + sample_label_0[:int(rate*count_label_0)]
        else:
            sample_label = sample_label + sample_label_0
            rate = count_label_0/count_label_1
            while(rate > 1):
                sample_label = sample_label + sample_label_1
                rate -= 1
            if rate > 0:
                sample_label = sample_label + sample_label_0[:int(rate*count_label_1)]

        random.shuffle(sample_label)

        # k fold
        assert k > 0
        len_per_fold = len(sample_label)//k
        sample_k_fold = []
        for i in range(k):
            if i != k-1:
                sample_k_fold.append(sample_label[i*len_per_fold:(i+1)*len_per_fold])
            else:
                sample_k_fold.append(sample_label[i*len_per_fold:])

        # write to file
        train_name = 'train'
        dev_name = 'dev'
        for i in range(k):
            temp_train_name = train_name + '_' + str(i+1)
            temp_dev_name = dev_name + '_' + str(i+1)
            temp_train_path = join(cfg.dataset_dir, temp_train_name)
            temp_dev_path = join(cfg.dataset_dir, temp_dev_name)
            with open(temp_train_path, 'w', encoding='utf-8') as f:
                for j in range(k):
                    if j==i:
                        continue
                    else:
                        f.writelines(sample_k_fold[j])
            with open(temp_dev_path, 'w', encoding='utf-8') as f:
                f.writelines(sample_k_fold[i])

    # upsampling, apply upsampling only in train not in dev, dev don't have repeative data in train
    def upsampling_1(self, k):
        _logger.add()
        _logger.add('upsampling ...')
        sample_label = []
        sample_label_0 = []
        sample_label_1 = []
        count_label_0 = 0
        count_label_1 = 0
        with open(self.en_train_file, 'r', encoding='utf-8') as f:
            for id,line in enumerate(f):
                if len(line) < 2:  # empty line
                    continue
                line_spl = line.split()
                if line[-1] != '\n': # last line
                    line += '\n'
                if int(line_spl[-1]) == 0:
                    count_label_0 += 1
                    sample_label_0.append(line)
                elif int(line_spl[-1]) == 1:
                    count_label_1 += 1
                    sample_label_1.append(line)

        random.shuffle(sample_label_0)
        random.shuffle(sample_label_1)


        for i in range(k):
            random.shuffle(sample_label_0)
            random.shuffle(sample_label_1)
            # k fold , generate train, and dev
            dev_sample_label_0 = sample_label_0[:count_label_0//k]
            train_sample_label_0 = sample_label_0[count_label_0//k:]
            dev_sample_label_1 = sample_label_1[:count_label_1//k]
            train_sample_label_1 = sample_label_1[count_label_1//k:]
            dev_sample = dev_sample_label_0 + dev_sample_label_1
            random.shuffle(dev_sample)

            # upsampling for train dataset
            train_count_label_0 = len(train_sample_label_0)
            train_count_label_1 = len(train_sample_label_1)
            train_sample = []
            if train_count_label_0 < train_count_label_1:
                rate = train_count_label_1 / train_count_label_0
                train_sample = train_sample + train_sample_label_1
                while(rate > 1):
                    train_sample = train_sample + train_sample_label_0
                    rate -= 1
                if(rate > 0):
                    train_sample = train_sample + train_sample_label_0[:int(rate*train_count_label_0)]
            else:
                rate = train_count_label_0 / train_count_label_1
                train_sample = train_sample + train_sample_label_0
                while(rate > 1):
                    train_sample = train_sample + train_sample_label_1
                    rate -= 1
                if (rate >0):
                    train_sample = train_sample + train_sample_label_1[:int(rate * train_count_label_1)]
            random.shuffle(train_sample)

            # write to file
            train_name = 'train'
            dev_name = 'dev'
            temp_train_name = train_name + '_' + str(i+1)
            temp_dev_name = dev_name + '_' + str(i+1)
            temp_train_path = join(cfg.dataset_dir, temp_train_name)
            temp_dev_path = join(cfg.dataset_dir, temp_dev_name)
            with open(temp_train_path, 'w', encoding='utf-8') as f:
                f.writelines(train_sample)
            with open(temp_dev_path, 'w', encoding='utf-8') as f:
                f.writelines(dev_sample)

#use nltk toolkit to tokenize sentence
def tokenize(text, language='en'):
    if language == 'en':
        return nltk.word_tokenize(text)
    elif language == 'es':
        sepecial_token1 = re.compile(r'\¿')
        sepecial_token2 = re.compile(r'\¡')
        text = sepecial_token1.sub(' ¿ ', text)
        text = sepecial_token2.sub(' ¡ ', text)
        return nltk.word_tokenize(text, 'spanish')
    else:
        return nltk.word_tokenize(text)