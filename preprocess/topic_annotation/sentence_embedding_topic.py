from sentence_transformers import SentenceTransformer
import torch
import jsonlines
from C99 import C99
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

import json
from tqdm import tqdm
import pickle

def encode_conversation(profix):
    data = []
    with open('data/dialogsum.'+profix + '.jsonl', encoding = 'utf8') as json_file:
        data_ = jsonlines.Reader(json_file)
        for obj in data_:
            data.append(obj)
    
    sent = []
    for i in range(0, len(data)):
        if len(data[i]['dialogue'].split('\r\n')) > 1:
            sentences = data[i]['dialogue'].split('\r\n')
        else:
            sentences = data[i]['dialogue'].split('\n')
        sent.append(sentences)
        
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sent))):
            #tokens_input = tokenize_conv(sent[i]).cuda()
            embedding = embedder.encode(sent[i])

            embeddings.append(embedding)
            
    with open('data/'+profix + '_sentence_transformer.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
        
    return embeddings


embeddings = encode_conversation('train')

def encode_convs(profix):
    model = C99(window = 4, std_coeff = 1)
    sent_label = []
    with open(profix+"_sentence_transformer.pkl", 'rb') as f:
        data = pickle.load(f)
    for i in range(0, len(data)):
        boundary = model.segment(data[i])
        temp_labels = []
        l = 0
        for j in range(0, len(boundary)):
            if boundary[j] == 1:
                l += 1
            temp_labels.append(l)
        sent_label.append(temp_labels)

        # print(temp_labels)
        # print(test_context[i])
        # print(raw_triples[i])
        # input()
    
    with open(profix + '_sent_c99_label.pkl', 'wb') as f:
        pickle.dump(sent_label, f)
    
    return sent_label

C99_labels = encode_convs('data/train')