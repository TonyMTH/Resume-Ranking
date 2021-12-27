import glob
import re

import numpy as np
from num2words import num2words
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tika import parser
import textract
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from torch.utils.data import Dataset

from training.parameters import column_mapping

nltk.download('stopwords')
nltk.download('punkt')


class Utils:

    def __init__(self):
        """
        Preprocesses document text queries
        """
        self.porter = PorterStemmer()
        self.stopwords = stopwords.words('english')

    def extract_text_from_resume(self, file_name):
        """
        Converts documents to text
        :param file_name: str
        :return: str
        """
        if file_name.split('.')[-1] == "pdf":
            text = parser.from_file(file_name)['content']
        elif file_name.split('.')[-1] in ["doc", "docx", "txt", "DOCX"]:
            text = textract.process(file_name).decode()
        else:
            text = ""
        return text

    def process_single_document(self, doc):
        """
        preprocess single document
        :param doc: str
        :return: list(str)
        """
        alpn = re.sub(r'[^a-zA-Z0-9]', ' ', doc).lower()
        tokens = nltk.word_tokenize(alpn)

        filtered_words = [self.porter.stem(word) for word in tokens if word not in self.stopwords]
        words_without_nums = []
        for word in filtered_words:
            try:
                word = num2words(word)
            except:
                pass
            words_without_nums.append(word)
        return words_without_nums

    def extract_keys_from_categories(self, path_to_categories, categories):
        """
        Assign keys to each data category
        :param path_to_categories: str
        :param categories: list
        :return: tuple(dict,dict,dict,dict)
        """

        cat_ids = dict((i, j) for i, j in enumerate(categories))
        cat_ids_inv = dict((j, i) for i, j in enumerate(categories))

        doc_dic = {}
        query_dic = {}

        for c_id, cat in enumerate(categories):
            fil_id = 0
            for file in glob.glob(path_to_categories + cat + "/*"):
                if not file.startswith('~'):
                    try:
                        text = self.extract_text_from_resume(file)
                        if 'query' in file.split('/')[-1]:
                            query_dic[c_id] = self.process_single_document(text)
                        else:
                            doc_id = str(c_id) + ':' + str(fil_id)
                            doc_dic[doc_id] = self.process_single_document(text)
                            fil_id += 1
                    except:
                        pass
        return cat_ids, cat_ids_inv, query_dic, doc_dic

    def process_resumes(self, path_to_categories, categories, scores, query_name, feature_name):
        """
        Apply extract_keys_from_categories to all resumes
        :param path_to_categories: str
        :param categories: list
        :param scores: dict
        :param query_name: str
        :param feature_name: str
        :return: pandas dataframe
        """
        cat_ids, cat_ids_inv, query_dic, doc_dic = self.extract_keys_from_categories(path_to_categories, categories)
        all_collection = []

        for key, val in scores.items():
            for i, v in enumerate(val):
                [all_collection.append([v, cat_ids_inv[key], list(set(query_dic[cat_ids_inv[key]])), doc_dic[j]])
                 for j in doc_dic.keys() if int(j.split(':')[0]) == i]

        return pd.DataFrame(all_collection, index=None, columns=['y', 'qid', query_name, feature_name])

    def clean_data(self, X):
        return [float(x.split(":")[-1]) if not isinstance(x, int) and not isinstance(x, float) else x for x in X]

    def clean_save_data(self, train, test, valid, cols, out_path):
        df_train = pd.read_csv(train, sep=" ", header=None)
        df_train = df_train[cols].apply(lambda x: self.clean_data(x)).rename(columns=column_mapping)

        df_test = pd.read_csv(test, sep=" ", header=None)
        df_test = df_test[cols].apply(lambda x: self.clean_data(x)).rename(columns=column_mapping)

        df_valid = pd.read_csv(valid, sep=" ", header=None)
        df_valid = df_valid[cols].apply(lambda x: self.clean_data(x)).rename(columns=column_mapping)

        df = df_train.append(df_test, ignore_index=True).append(df_valid, ignore_index=True)

        df.to_csv(out_path, index=False)

    def load_data(self, path):
        return pd.read_csv(path)

    def split_data(self, data, valid_data, test_size=.1):
        qid_index, qid_valid_index = data.columns.get_loc("qid"), valid_data.columns.get_loc("qid")
        x, y = data.drop(['y'], axis=1).values, data.y.values
        x_valid, y_valid, qid_valid = valid_data.drop(['y', 'qid'],
                                                      axis=1).values, valid_data.y.values, valid_data.qid.values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=0)
        qid_train, qid_test = x_train[:, qid_index], x_test[:, qid_index]
        x_train = np.delete(x_train, qid_index, axis=1)
        x_test = np.delete(x_test, qid_index, axis=1)
        return self.normalize(x_train), self.normalize(x_test), self.normalize(x_valid), y_train, y_test, y_valid,\
               qid_train, qid_test, qid_valid

    def normalize(self, x):
        scalar = StandardScaler()
        return scalar.fit_transform(x)


class Datasets(Dataset):
    def __init__(self, label, features, qid):
        self.qid = qid
        self.labels = label
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        return self.qid[i], self.labels[i], self.features[i]



if __name__ == '__main__':
    pth = 'data/'
    data_path = '../data/vali.txt'
    utils = Utils()
    train = 'data/Fold1/vali.txt'
    df = utils.clean_save_data(data_path)

    # df = processTrainResume.required_features(df, required_columns)

    # df = processTrainResume.process_resumes(pth, categories, scores, query_name, feature_name)
    # df = processTrainResume.generate_features(df, query_name, feature_name)
    print(df.head())
