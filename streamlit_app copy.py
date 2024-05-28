import streamlit as st
from google.cloud import firestore
import pyrebase
import json
import fitz
from utils import *
import pickle
from datetime import datetime
import uuid
from utils import *
import os
from tqdm import tqdm


# Authenticate to Firestore with the JSON account key.
#db = firestore.Client.from_service_account_json("robotofficer_firebase.json")

# Create a reference to the Google post.
#doc_ref = db.collection("users").document("user")

# Then get the data at that reference.
#doc = doc_ref.get()

# Let's see what we got!
#st.write("The id is: ", doc.id)
#st.write("The contents are: ", doc.to_dict())


#doc_ref_2 = db.collection("users").document("user_2")
#doc_ref_2.set({
#	"user_email": "456@123.com",
#	"user_age": "35",
#	"user_id": "2"
#})
#doc2 = doc_ref_2.get()
#st.write("The contents are: ", doc2.to_dict())

#users_ref = db.collection("users")

# For a reference to a collection, we use .stream() instead of .get()
#for doc in users_ref.stream():
#    st.write("The id is: ", doc.id)
#    st.write("The contents are: ", doc.to_dict())

def get_model_summary_vectorize(sentence):
    result = []
    for model_folder in model_folders:
        try:
            fine_tune_sentiment_model = AutoModelForSequenceClassification.from_pretrained('MikeZQZ/%s'%model_folder,
                                                                                                num_labels=2)
            fine_tune_tokenizer = AutoTokenizer.from_pretrained('MikeZQZ/%s'%model_folder)
            fine_tune_pipeline = TextClassificationPipeline(model=fine_tune_sentiment_model, 
                                                                    tokenizer = fine_tune_tokenizer)
            #print("%s result: "%model_folder , fine_tune_pipeline.predict('I like python')[0]['label'].split('_')[-1])
            rlt = int(fine_tune_pipeline.predict(sentence)[0]['label'].split('_')[-1])
            if rlt == 1:
                result.append(label_dict[model_folder])
        except:
            #print("%s not exists because no enough data"%model_folder)
            result.append("no enough data for topic : %s"%label_dict[model_folder])
    return '|'.join(result)

def get_model_pred(sentence, model_folders):
    result = []
    for model_folder in model_folders:
        try:
            fine_tune_sentiment_model = AutoModelForSequenceClassification.from_pretrained('MikeZQZ/%s'%model_folder,
                                                                                                num_labels=2)
            fine_tune_tokenizer = AutoTokenizer.from_pretrained('MikeZQZ/%s'%model_folder)
            fine_tune_pipeline = TextClassificationPipeline(model=fine_tune_sentiment_model, 
                                                                    tokenizer = fine_tune_tokenizer)
            #print("%s result: "%model_folder , fine_tune_pipeline.predict('I like python')[0]['label'].split('_')[-1])
            rlt = int(fine_tune_pipeline.predict(sentence)[0]['label'].split('_')[-1])
            if rlt == 1:
                result.append(label_dict[model_folder])
        except Exception as e: 
            #print(e)
            #print("%s not exists because no enough data"%model_folder)
            result.append("no enough data for topic : %s"%label_dict[model_folder])
    return '|'.join(result)


def get_firebase_folder_name(filename):
     #### file store destination
    cur_filename = str(filename.split(".pdf")[0]).strip()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    cur_uuid = str(uuid.uuid4().hex)
    cur_uuid_str = "%s%s"%(cur_uuid[:3], cur_uuid[-3:])
    cur_result_folder = "%s_%s_%s"%(cur_filename,cur_uuid_str,timestamp)
    cur_cloud_folder = "results/%s/"%cur_result_folder
    return cur_cloud_folder

#### load config for firebase
with open("robotofficer_firebase_storage.json","rb") as f:
    file_config = json.load(f)
filedb = pyrebase.initialize_app(file_config)
storage = filedb.storage()

#### load local model files
dict_files = ["label_col_dict_p1.pkl","label_col_dict_p2_new.pkl","label_col_dict_p3.pkl"]
label_dict = get_topic_dict(dict_files)
model_folders = list(label_dict.keys())
#model_folders = ['label_1','label_0']

#### start to upload pdf and analysis
uploaded_pdf = st.file_uploader("Upload your file (PDF only): ", type=['pdf'])
if uploaded_pdf is not None:
    with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            try:
                local_pdf_file = uploaded_pdf.name
                doc.save(local_pdf_file)
                cur_pdf = get_all_sentence_by_file(uploaded_pdf.name, doc)
                print("%s has been sliced"%local_pdf_file)
                #v_get_model_summary = np.vectorize(get_model_summary_vectorize)
                cur_pdf = cur_pdf[cur_pdf['sentence'].astype(str).map(len)>=10].reset_index(drop=True)
                sentences = list(cur_pdf['sentence'].values)
                cur_pdf_result_ls = []
                for sentence in tqdm(sentences):
                    cur_pdf_result_ls.append(get_model_pred(sentence, model_folders))
                #cur_pdf_result = pd.DataFrame(v_get_model_summary(cur_pdf['sentence']), 
                #                               columns = ['topic_overall_result'])
                cur_pdf_result = pd.DataFrame(np.array(cur_pdf_result_ls),
                                               columns = ['topic_overall_result']).reset_index(drop=True)
                cur_pdf_result['topic_hitted'] = cur_pdf_result['topic_overall_result']\
                    .apply(lambda x : [i for i in x.split("|")
                                        if ('no enough data for topic' not in i) and len(i)>0])
                cur_df = pd.concat([cur_pdf, cur_pdf_result], axis=1).explode('topic_hitted')
                local_pdf_results = "%s_results.csv"%(local_pdf_file.split(".pdf")[0].strip())
                cur_df.to_csv(local_pdf_results)
                cur_cloud_folder = get_firebase_folder_name(local_pdf_file)
                storage.child("%s%s"%(cur_cloud_folder, local_pdf_file)).put(local_pdf_file)
                storage.child("%s%s"%(cur_cloud_folder, local_pdf_results)).put(local_pdf_results)
                #print(cur_df.explode('topic_hitted'))
                st.write(cur_df)
                os.remove(local_pdf_file)
                os.remove(local_pdf_results)
            except:
                 print("something wrong with the pdf file or the model prediction")