# About data collection
import torch
from torch import nn, optim

pth = '../data/'
data_path = '../data/train.txt'
data_train_path = '../data/train.txt'
data_test_path = '../data/test.txt'
data_valid_path = '../data/vali.txt'
clean_data_path = '../data/clean_data.csv'
resume_path = '../production/data/resume.csv'

categories = ['java', 'data', 'admin', 'analyst', 'legal', 'account']
scores = {'java': [4, 2, 0, 1, 0, 0],
          'data': [2, 4, 1, 2, 0, 0],
          'admin': [1, 1, 4, 1, 1, 1],
          'analyst': [0, 2, 1, 4, 0, 0],
          'legal': [0, 1, 2, 1, 4, 0],
          'account': [0, 1, 2, 1, 0, 4]}
required_columns = [0, 1, 16, 21, 26, 31, 36, 41, 46, 76, 81, 86, 91, 96, 106, 111]
column_names = ['y', 'doc_len', 'idf', 'sum_tf', 'min_tf', 'max_tf', 'mean_tf', 'var_tf', 'sum_tfidf', 'min_tfidf',
                'max_tfidf', 'mean_tfidf', 'var_tfidf', 'cosine', 'bm25']
column_mapping = {0: 'y', 26: 'sum_tf', 31: 'min_tf', 36: 'max_tf', 41: 'mean_tf', 46: 'var_tf', 21: 'idf',
                  76: 'sum_tfidf', 81: 'min_tfidf', 86: 'max_tfidf', 91: 'mean_tfidf', 96: 'var_tfidf',
                  16: 'doc_len', 111: 'bm25', 106: 'cosine', 1: 'qid'}
query_name, feature_name = 'query', 'feature'

# Training
lr = 0.001
criterion = nn.MSELoss()
Optimizer = lambda x: optim.Adam(x, lr=lr)
scheduler = lambda x: torch.optim.lr_scheduler.StepLR(x, 1, gamma=0.95)
hidden1, hidden2, hidden3 = 128, 256, 64

epochs = 10
printing_gap = 1
model_path = '../production/data/best_model.pt'
reg_model_path = '../production/data/reg_model.pt'
svm_model_path = '../production/data/svm_model.pt'
saved_model_device = torch.device("cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

k_rank = 10
PIK_plot_data = '../data/plot_data.dat'
