import fitz
import streamlit as st
from utils import *
import pickle
from datetime import datetime
import uuid
import json
import pyrebase
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer
from firebase_admin import credentials, initialize_app, storage

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
cur_uuid = str(uuid.uuid4().hex)

cur_result_folder = "%s_%s"%(timestamp, cur_uuid)

cur_cloud_folder = "results/%s"%cur_result_folder
#def get_topic_models():
     ###tbd
#     print(123)

with open("robotofficer_firebase_storage.json","rb") as f:
    file_config = json.load(f)
filedb = pyrebase.initialize_app(file_config)
storage = filedb.storage()


import firebase_admin
from firebase_admin import credentials, initialize_app, storage

cred = credentials.Certificate("robotofficer-8db01-firebase-adminsdk-rkpmg-538352a0bd.json")
firebase_admin.initialize_app(cred)
bucket = storage.bucket("robotofficer-8db01.appspot.com")
for i in bucket.list_blobs(prefix='topic_models/label_10/final_model'):
    print(i.name)

#print(storage.child("topic_models").list_files())
model = AutoModelForSequenceClassification.from_pretrained('MikeZQZ/label_0')

#fine_tune_sentiment_model = AutoModelForSequenceClassification.from_pretrained('%s/final_model/'%model_folder,
 #                                                                                               num_labels=2)

#get_model_summary_vectorize(sentence, model_folders, label_dict)
'''
uploaded_pdf = st.file_uploader("Upload your file (PDF only): ", type=['pdf'])
if uploaded_pdf is not None:
    print(uploaded_pdf.name) 
    with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            cur_df = get_all_sentence_by_file(uploaded_pdf.name,doc)
            print(cur_df)
            st.dataframe(cur_df)
            cur_df_ = cur_df[cur_df['sentence'].astype(str).map(len)>=10]
            v_get_model_summary = np.vectorize(get_model_summary_vectorize)
'''