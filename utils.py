import os
from docx import Document
import textract
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tika import parser


def tf(docs):
    cv = CountVectorizer()
    return cv.fit_transform(docs)

#tf = pd.DataFrame(tf(docs).toarray(), columns=cv.get_feature_names())
#print(tf)

path = 'data/resume_data/'
for job_type in os.listdir(path):
    arr = os.listdir(path+job_type)
    #print(arr)


# document = Document('data/resume_data/account/1.doc')
# for para in document.paragraphs:
#     print(para.text)

# text = textract.process(r'C:\Users\Asus\Documents\Education\Courses\Strive School\Projects\Resume-Ranking\data\resume_data\account\0.doc')
# print(text)


from glob import glob
import re
import os
import win32com.client as win32
from win32com.client import constants

# Create list of paths to .doc files
p='C:\\Users\\Asus\\Documents\\Education\\Courses\\Strive School\\Projects\\Resume-Ranking\\data\\resume_data'
paths = glob(p+'\\**\\*.doc', recursive=True)

def save_as_docx(path):
    # Opening MS Word
    word = win32.gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(path)
    doc.Activate ()

    # Rename path with .docx
    new_file_abs = os.path.abspath(path)
    new_file_abs = re.sub(r'\.\w+$', ' new.docx', new_file_abs)

    # Save and Close
    word.ActiveDocument.SaveAs(
        new_file_abs, FileFormat=constants.wdFormatXMLDocument
    )
    doc.Close(False)

for path in paths:
    # if '~' in path:
    # print(path)
    #save_as_docx(path)
    pass

def extract_text_from_resume(file_name):
    """
    Converts documents to text
    :param file_name: str
    :return: str
    """
    if file_name.split('.')[-1] == "pdf":
        text = parser.from_file(file_name)['content']
    elif file_name.split('.')[-1] in ["doc", "docx", "txt"]:
        text = textract.process(file_name).decode()
    else:
        text = ""
    return text
file_name = 'data/resume_data/account/1.doc'
extract_text_from_resume(file_name)