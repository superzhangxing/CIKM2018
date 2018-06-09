from  configs import cfg
import random
from src.record_log import _logger
from os.path import join

class PreprocessData(object):
    def __init__(self, org_trian_file):
        self.org_train_file = org_trian_file

    # k fold
    def downsampling(self, k):
        sample_label = []
        sample_label_0 = []
        sample_label_1 = []
        count_label_0 = 0
        count_label_1 = 0
        with open(self.org_train_file, 'r', encoding='utf-8') as f:
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




