# Import libraries
import pickle

import torch
from torch.utils.data import DataLoader

from feature_extraction import FeatureExtraction
from utils import Datasets, Utils

torch.manual_seed(0)


class Rank:
    def __init__(self):
        self.model_path = './data/best_model.pt'
        self.rfr_model_path = './data/reg_model.pt'
        self.svm_model_path = './data/svm_model.pt'

    def predict(self, query):
        # Load Model
        model = torch.load(self.model_path)
        w = list(model.parameters())
        model.eval()

        utils = Utils()
        qid, data_tfidf, data_bm25, data = utils.process(query, "./data/resume_location.txt")
        dataset = Datasets(qid, data)

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        tfidf = [(q, d) for d, q in
                 sorted(zip(data_tfidf.tolist(), qid.tolist()), key=lambda pair: pair[0], reverse=True)]
        bm25 = [(q, d) for d, q in
                sorted(zip(data_bm25.tolist(), qid.tolist()), key=lambda pair: pair[0], reverse=True)]

        with open(self.rfr_model_path, "rb") as f:
            rfr = pickle.load(f)
        rfr = rfr.predict(data)
        rfr = [(q, d) for d, q in sorted(zip(rfr, qid), key=lambda pair: pair[0], reverse=True)]

        # with open(self.svm_model_path, "rb") as f:
        #     svm = pickle.load(f)
        # svm = svm.predict(data)
        # svm = [(q, d) for d, q in sorted(zip(svm, qid), key=lambda pair: pair[0], reverse=True)]
        svm = [(q, d) for d, q in sorted(zip(rfr, qid), key=lambda pair: pair[0], reverse=True)]

        nn, idq = [], []
        for qid, features in loader:
            features = features.float()
            model.to("cpu")
            output = model(features)
            nn.append(output.item())
            idq.append(qid[0])

        nn = [(q, d) for d, q in sorted(zip(nn, idq), key=lambda pair: pair[0], reverse=True)]

        return tfidf, bm25, nn, rfr, svm


if __name__ == '__main__':
    model_pth = 'data/best_model.pt'
    out = Rank().predict('data')
    print(out)
