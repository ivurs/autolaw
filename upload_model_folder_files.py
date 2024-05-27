import streamlit as st
from google.cloud import firestore
import pyrebase
import json
import os
from tqdm import tqdm

with open("robotofficer_firebase_storage.json","rb") as f:
    file_config = json.load(f)
#print(file_config)
filedb = pyrebase.initialize_app(file_config)

all_topic_paths = [i for i in os.listdir("../autoreviewer") if os.path.isdir(os.path.join("../autoreviewer",i)) and 'label_' in i]
storage = filedb.storage()

for topic_path in tqdm(all_topic_paths):
    folder_path_on_cloud = 'topic_models/%s/final_model/'%topic_path
    folder_path_on_local = "../autoreviewer/%s/final_model/"%topic_path
    for i in tqdm(os.listdir(folder_path_on_local)):
        try:
            storage.child("%s%s"%(folder_path_on_cloud, i)).put("%s%s"%(folder_path_on_local,i))
            print("%s%s successfully uploaded"%(folder_path_on_local,i) )
        except:
            print("%s%s failed to upload"%(folder_path_on_local,i) )
            continue

'''

for i in os.listdir(folder_path_on_local):
    print(i)
    storage.child("%s%s"%(folder_path_on_cloud, i)).put("%s%s"%(folder_path_on_local,i))
'''