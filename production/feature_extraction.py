from collections import Counter
import numpy as np


class FeatureExtraction:
    def __init__(self):
        """
        Create input features
        """
        pass

    def corpus_words_dic(self, corpus):
        """
        Generate vocab of dict from all corpus in df
        :param corpus: list(list)
        :return: dict
        """
        d = {}
        for i, c in enumerate(corpus):
            for k, v in dict(Counter(c)).items():
                try:
                    d[k].append(i)
                except:
                    d[k] = [i]
        return d

    def tf(self, query, corpus):
        """
        Compute Term Frequency
        :param query: list(int)
        :param corpus: list(int)
        :return: list(int)
        """
        freq = []
        for t in query:
            t_count = corpus.count(t)
            # if t_count != 0:
            freq.append(t_count)
        return np.array(freq)

    def idf(self, query, corpus):
        """
        Compute Inverse Document Frequency
        :param query: list(int)
        :param corpus: list(int)
        :return: list(float)
        """
        corpus = self.corpus_words_dic(corpus)
        c = len(corpus)
        idf = []
        for q in query:
            try:
                df = len(corpus[q])
            except:
                df = 0
            idf.append(np.log10(1 + (c - df + 0.5) / (df + 0.5)))
        return np.array(idf)

    def tfidf(self, query, corpus):
        """
        Compute TF-IDF
        :param query: list(int)
        :param corpus: list(int)
        :return: list(float)
        """
        return self.tf(query, corpus) * self.idf(query, corpus)

    def average_doc_len(self, df_feature):
        """
        Get the average word lengths for all documents in the corpus
        :param df_feature: array
        :return: float
        """
        avgdl = []
        for d in df_feature:
            cor = self.corpus_words_dic(d)
            avgdl.append(len(cor))
        return np.mean(avgdl)

    def bm25(self, query, corpus, avgdl, k1=2.5, k3=0, b=0.8):
        """
        Compute BM25 for a given query and corpus
        :param query: str
        :param corpus: str
        :param avgdl: float
        :param k1: float or int
        :param k3: float or int
        :param b: float or int
        :return: float
        """
        cor = self.corpus_words_dic(corpus)
        c = len(cor)
        pt1 = (self.tf(query, corpus) * self.idf(query, corpus) * (k1 + 1)) / (
                self.tf(query, corpus) + k1 * (1 - b + b * (c / avgdl)))
        pt2 = ((k3 + 1) * self.tf(query, query)) / (k3 + self.tf(query, query))
        return sum(pt1 * pt2)

    def dot_product(self, query, corpus):
        """
        Dot product between arrays
        :param query: str
        :param corpus: str
        :return: float
        """
        query = Counter(query)
        corpus = Counter(corpus)
        s = 0.0
        for key in set(query):
            if key in corpus:
                s += (query[key] * corpus[key])
        return s

    def cosine_similarity(self, query, corpus):
        """
        Cosine similarity
        :param query:
        :param corpus:
        :return: float
        """
        num = self.dot_product(query, corpus)
        den = np.sqrt(self.dot_product(query, query) * self.dot_product(corpus, corpus))

        return np.arccos(num / den)

    def generate_features(self, df, query):
        # Term Frequencies
        df['tf'] = df['text'].apply(lambda x: self.tf(query, x))  # drop
        df['sum_tf'] = df['tf'].apply(np.sum)
        df['min_tf'] = df['tf'].apply(np.min)
        df['max_tf'] = df['tf'].apply(np.max)
        df['mean_tf'] = df['tf'].apply(np.mean)
        df['var_tf'] = df['tf'].apply(np.var)

        # Inverse Term Frequency
        df['idf'] = df['text'].apply(lambda x: self.idf(query, x)).apply(np.sum)

        # TF-IDF
        df['tfidf'] = df['text'].apply(lambda x: self.tfidf(query, x))  # drop
        df['sum_tfidf'] = df['tfidf'].apply(np.sum)
        df['min_tfidf'] = df['tfidf'].apply(np.min)
        df['max_tfidf'] = df['tfidf'].apply(np.max)
        df['mean_tfidf'] = df['tfidf'].apply(np.mean)
        df['var_tfidf'] = df['tfidf'].apply(np.var)

        # Document length
        df['doc_len'] = df['text'].apply(lambda x: len(x))

        # BM25
        avgdl = self.average_doc_len(df['text'].values)
        df['bm25'] = df['text'].apply(lambda x: self.bm25(query, x, avgdl))

        # Cosine Similarity
        df['cosine'] = df['text'].apply(lambda x: self.cosine_similarity(query, x))

        df.drop(['tf', 'tfidf', 'text'], axis=1, inplace=True)

        return df


if __name__ == '__main__':
    from utils import Utils
    import parameters as para

    utils = Utils()
    featureExtraction = FeatureExtraction()

    resume = utils.process_resumes(para.pth, para.categories, para.scores, para.query_name, para.feature_name)
    featureExtraction.generate_features(resume, para.query_name, para.feature_name, para.resume_path)

    utils.clean_save_data(para.data_path, para.required_columns, para.clean_data_path)

