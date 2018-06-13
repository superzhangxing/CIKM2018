from configs import cfg
from src.record_log import _logger
import numpy as np
from tqdm import tqdm
import random
import math
import re
import os
import io
import nltk
from tokenize import  generate_tokens

from src.file import save_file

PADDING = '<pad>'
UNKNOWN = '<unk>'

class Dataset(object):
    def __init__(self, data_file_path, data_type, dicts=None, language_type = 'es', unlabeled_file_path=None, emb_file_path=None):
        self.data_type = data_type
        _logger.add('building data set object for %s' % data_type)
        assert data_type in ['train', 'dev', 'test','infer']

        # check
        if data_type in ['dev', 'test','infer']:
            assert dicts is not None

        # build vocab  --> deprecated --> dicts is preprocessed
        # if data_type == 'train':
        #     assert unlabeled_file_path is not None
        #     self.dicts = {}
        #     vocab_es_count, vocab_es_token2id, vocab_es_id2token, vocab_en_count, vocab_en_token2id, vocab_en_id2token=build_vocab(unlabeled_file_path)
        #     self.dicts['es'] = vocab_es_token2id
        #     self.dicts['en'] = vocab_en_token2id
        # else:
        #     self.dicts = dicts
        self.dicts = dicts

        # data
        if data_type=='train':
            self.nn_data = load_en_train_data(data_file_path, self.dicts['es'], self.dicts['en'])
        elif data_type=='dev':
            self.nn_data = load_en_train_data(data_file_path, self.dicts['es'], self.dicts['en'])
        elif data_type == 'test':
            self.nn_data = load_es_train_data(data_file_path, self.dicts['es'], self.dicts['en'])
        elif data_type == 'infer':
            self.nn_data = load_es_test_data(data_file_path, self.dicts['es'])
        self.sample_num = len(self.nn_data)
        # generate embedding
        if data_type == 'train':
            if language_type == 'en':
                self.emb_mat_token = load_emb_mat(emb_file_path, self.dicts['en'])
            elif language_type == 'es':
                self.emb_mat_token = load_emb_mat(emb_file_path, self.dicts['es'])



    # ------------------ internal use --------------------------------
    def save_dict(self, path):
        save_file(self.dicts, path, 'token dict data', 'pickle')

    def filter_data(self, data_type=None):
        # TODO
        pass

    def generate_batch_sample_iter(self, max_step=None):
        if max_step is not None:
            # for train, with max_step and batch_size
            batch_size = cfg.train_batch_size

            def data_queue(data,batch_size):
                assert len(data) >= batch_size
                random.shuffle(data)
                data_ptr = 0
                dataRound = 0
                idx_b = 0
                step = 0
                while True:
                    if data_ptr + batch_size <= len(data):
                        yield data[data_ptr:data_ptr + batch_size], dataRound, idx_b
                        data_ptr += batch_size
                        idx_b += 1
                        step += 1
                    elif data_ptr + batch_size > len(data):
                        offset = data_ptr + batch_size - len(data)
                        out = data[data_ptr:]
                        random.shuffle(data)
                        out += data[:offset]
                        data_ptr = offset
                        dataRound += 1
                        yield out, dataRound, 0
                        idx_b = 1
                        step += 1
                    if step >= max_step:
                        break

            batch_num = math.ceil(len(self.nn_data) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(self.nn_data, batch_size):
                yield sample_batch,batch_num,data_round,idx_b
        else:
            # for test, only 1 data_round,
            batch_size = cfg.test_batch_size
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in self.nn_data:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b


# ---------------------- internal use ------------------------
# # build vocab with unlabeled data
def build_vocab(es_en_file):
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
    with open(es_en_file, 'r', encoding='utf-8') as f:
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

    return vocab_es_count,vocab_es_token2id,vocab_es_id2token,vocab_en_count,vocab_en_token2id,vocab_en_id2token

def load_es_train_data(file, vocab_es_token2id, vocab_en_token2id):
    """
    :param file: type--> sent1_es \t sent1_en \t sent2_es \t sent2_en \t label
    :return:
    """
    samples = []
    with open(file, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sample = {}
            line_split = line.split('\t')
            # sent1_token_es = line_split[0].split()
            # sent1_token_en = line_split[1].split()
            # sent2_token_es = line_split[2].split()
            # sent2_token_en = line_split[3].split()
            sent1_token_es = tokenize(line_split[0], 'es')
            sent1_token_en = tokenize(line_split[1], 'en')
            sent2_token_es = tokenize(line_split[2], 'es')
            sent2_token_en = tokenize(line_split[3], 'en')
            label = int(line_split[4])
            if len(line_split) != 5:  # check
                continue
            if cfg.lower_word:
                sample['sent1_token_id_es'] = [vocab_es_token2id.get(token.lower(), 1) for token in sent1_token_es]
                sample['sent1_token_id_en'] = [vocab_en_token2id.get(token.lower(), 1) for token in sent1_token_en]
                sample['sent2_token_id_es'] = [vocab_es_token2id.get(token.lower(), 1) for token in sent2_token_es]
                sample['sent2_token_id_en'] = [vocab_en_token2id.get(token.lower(), 1) for token in sent2_token_en]
            else:
                sample['sent1_token_id_es'] = [vocab_es_token2id.get(token, 1) for token in sent1_token_es]
                sample['sent1_token_id_en'] = [vocab_en_token2id.get(token, 1) for token in sent1_token_en]
                sample['sent2_token_id_es'] = [vocab_es_token2id.get(token, 1) for token in sent2_token_es]
                sample['sent2_token_id_en'] = [vocab_en_token2id.get(token, 1) for token in sent2_token_en]
            sample['label'] = label
            samples.append(sample)
    return samples

def load_en_train_data(file, vocab_es_token2id, vocab_en_token2id):
    """
    :param file: type--> sent1_en \t sent1_es \t sent2_en \t sent2_es \t label
    :return:
    """
    samples = []
    with open(file, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sample = {}
            line_split = line.split('\t')
            # sent1_token_en = line_split[0].split()
            # sent1_token_es = line_split[1].split()
            # sent2_token_en = line_split[2].split()
            # sent2_token_es = line_split[3].split()
            sent1_token_en = tokenize(line_split[0], 'en')
            sent1_token_es = tokenize(line_split[1], 'es')
            sent2_token_en = tokenize(line_split[2], 'en')
            sent2_token_es = tokenize(line_split[3], 'es')
            label = int(line_split[4])
            if len(line_split) != 5:  # check
                continue
            if cfg.lower_word:
                sample['sent1_token_id_es'] = [vocab_es_token2id.get(token.lower(), 1) for token in sent1_token_es]
                sample['sent1_token_id_en'] = [vocab_en_token2id.get(token.lower(), 1) for token in sent1_token_en]
                sample['sent2_token_id_es'] = [vocab_es_token2id.get(token.lower(), 1) for token in sent2_token_es]
                sample['sent2_token_id_en'] = [vocab_en_token2id.get(token.lower(), 1) for token in sent2_token_en]
            else:
                sample['sent1_token_id_es'] = [vocab_es_token2id.get(token, 1) for token in sent1_token_es]
                sample['sent1_token_id_en'] = [vocab_en_token2id.get(token, 1) for token in sent1_token_en]
                sample['sent2_token_id_es'] = [vocab_es_token2id.get(token, 1) for token in sent2_token_es]
                sample['sent2_token_id_en'] = [vocab_en_token2id.get(token, 1) for token in sent2_token_en]
            sample['label'] = label
            samples.append(sample)
    return samples

def load_es_test_data(file, vocab_es_token2id):
    """

    :param file: type--> sent1_es \t sent2_es
    :param vocab_es_token2id:
    :return:
    """
    samples = []
    with open(file, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sample = {}
            line_split = line.split('\t')
            # sent1_token_es = line_split[0].split()
            # sent2_token_es = line_split[1].split()
            sent1_token_es = tokenize(line_split[0], 'es')
            sent2_token_es = tokenize(line_split[1], 'es')
            if len(line_split) != 2:  # check
                continue
            if cfg.lower_word:
                sample['sent1_token_id_es'] = [vocab_es_token2id.get(token.lower(), 1) for token in sent1_token_es]
                sample['sent2_token_id_es'] = [vocab_es_token2id.get(token.lower(), 1) for token in sent2_token_es]
            else:
                sample['sent1_token_id_es'] = [vocab_es_token2id.get(token, 1) for token in sent1_token_es]
                sample['sent2_token_id_es'] = [vocab_es_token2id.get(token, 1) for token in sent2_token_es]
            samples.append(sample)
    return samples

# generate embedding based on vocab and pretrained_dict_file
def load_emb_mat(file, vocab_token2id):
    emb_mat = np.random.uniform(-0.05,0.05, size=(len(vocab_token2id), cfg.word_embedding_length)).astype(cfg.floatX)
    with open(file, 'r', encoding='utf-8') as f:
        f.readline() # skip first line
        for line in f:
            tokens = line.rstrip().split()
            if len(tokens) > cfg.word_embedding_length:
                emb = tokens[len(tokens) - cfg.word_embedding_length :]
                token = ' '.join(tokens[:len(tokens) - cfg.word_embedding_length])
                if cfg.lower_word:
                    token = token.lower()
            if token in vocab_token2id:
                token2id = vocab_token2id[token]
                for i in range(cfg.word_embedding_length):
                    emb_mat[token2id][i] = float(emb[i])
    return emb_mat

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

# need to fix bug
# def tokenize(text, language='en'):
#     result = []
#     if len(text) == 0:
#         return result
#
#     for _,token,_,_,_ in generate_tokens(io.StringIO(text).readline):
#         result.append(token)
#     if len(result[-1]) == 0:
#         result = result[:-1]
#
#     return result
