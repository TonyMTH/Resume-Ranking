import glob
import re
import pandas as pd
from num2words import num2words
from sklearn.preprocessing import StandardScaler
from tika import parser
import textract
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from torch.utils.data import Dataset

from feature_extraction import FeatureExtraction

nltk.download('stopwords')
nltk.download('punkt')


class Utils:

    def __init__(self):
        """
        Preprocesses document text queries
        """
        self.porter = PorterStemmer()
        self.stopwords = stopwords.words('english')

    def iterate_resume(self, folder):
        all_file = []
        for file in glob.glob(folder + "/*"):
            try:
                text = self.extract_text_from_resume(file)
                # sent_text = nltk.sent_tokenize(text)
                title = file.split('/')[-1].split('.')[0].title()
                body = title + '\n' + '=' * 5 + '\n' + text[:50]
                all_file.append(body)
            except:
                pass
        return all_file

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
                word = num2words(word).split('-')
                words_without_nums.extend(word)
            except:
                words_without_nums.append(word)
        return words_without_nums

    def process_resumes(self, folder):
        all_file = []

        for file in glob.glob(folder + "/*"):
            try:
                text = self.extract_text_from_resume(file)
                tokens = self.process_single_document(text)
                key = file.split('/')[-1].split('.')[0]
                all_file.append([key, tokens])
            except:
                pass

        return pd.DataFrame(all_file, index=None, columns=['qid', 'text'])

    def process_query(self, text):
        tokens = self.process_single_document(text)
        return tokens

    def process(self, query, data_location):
        with open(data_location, "r") as text_file:
            dir = text_file.readlines()

        featureExtraction = FeatureExtraction()

        processed_resumes = self.process_resumes(dir[0].strip())
        processed_query = self.process_query(query)

        fe = featureExtraction.generate_features(processed_resumes, processed_query)

        return fe.qid.values, fe.mean_tfidf.values, fe.bm25.values, self.normalize(fe.drop(['qid'], axis=1).values)

    def normalize(self, x):
        scalar = StandardScaler()
        return scalar.fit_transform(x)


class Datasets(Dataset):
    def __init__(self, qid, features):
        self.qid = qid
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        return self.qid[i], self.features[i]


if __name__ == '__main__':
    pth = '/home/anthony/Documents/Strive/resume/production/data/resumes/'
    utils = Utils()
    df = utils.iterate_resume(pth)
