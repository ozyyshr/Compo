import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from tqdm import trange
import os
import matplotlib.pyplot as plt
import argparse
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def recognize_speaker(dialogue):
    speakers = []

    for uttr in dialogue:
        s = uttr.split(":")[0].strip()
        if s not in speakers and s != "<mask>":
            speakers.append(s)

    return speakers

def fast_votek(embeddings,select_num,k,vote_file=None):
    n = len(embeddings)

    # bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    for i in range(n):
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        for idx in sorted_indices:
            if idx!=i:
                vote_stat[idx].append(i)
        # bar.update(1)

    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices


def main():

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    num_clusters = 8
    random_seed = 27
    fix_seed(random_seed)

    num_da = 20000

    ori_data = []
    with open("./new_train.json") as f:
        for line in f:
            ori_data.append(json.loads(line))

    conv_pool = []
    with open('./conv_pool.json') as f:
        for line in f:
            conv_pool.append(json.loads(line))

    corpus = []
        
    for i in conv_pool:
        triples = i['triples']
        dialogue = i['dialogue']
        speakers = recognize_speaker(dialogue)

        sent = []

        for triple in triples:
            sub = triple[0]
            verb = triple[1]
            obj = triple[2]
            for si, speaker in enumerate(speakers):
                if speaker in sub:
                    sub = sub.replace(speaker, "<person_{}>".format(si))
                if speaker in verb:
                    verb = verb.replace(speaker, "<person_{}>".format(si))
                if speaker in obj:
                    obj = obj.replace(speaker, "<person_{}>".format(si))
            sent.append(sub + " " + verb + " " + obj + ".")
        
        corpus.append(" ".join(sent))
    
    # assert len(corpus) == len(ori_data)
    num_sample = len(ori_data)
                
    corpus_embeddings = encoder.encode(corpus)  # [num_conv, dim=384]

    # perform Kmeans clustering
    # clustering_model = KMeans(n_clusters=num_clusters, random_state=random_seed)
    # clustering_model.fit(corpus_embeddings)
    # cluster_assignment = clustering_model.labels_

    # for i in range(num_clusters):
    #     li = cluster_assignment.tolist()
    #     print(li.count(i))
    #     input()

    # vote-k algorithm
    # select_indices = fast_votek(embeddings=corpus_embeddings, select_num=5, k=150, vote_file=('./votek_cache.json'))
    # print(select_indices)
    # input()

    #  perform KNN search
    knn = NearestNeighbors(n_neighbors=100)
    knn.fit(corpus_embeddings)
    NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2, radius=1.0)
        
    # nns = knn.kneighbors(corpus_embeddings[0].reshape(1, -1), return_distance=False)

    new_samples = []
    for i in trange(num_da):
        random_num = random.choice(range(num_sample))
        actions = ori_data[random_num]['source'].split('</s>')[0]
        actions_embedding = encoder.encode(actions).reshape(1, -1)
        nns = knn.kneighbors(actions_embedding, return_distance=False)

        # vote-k method
        # select_indices = fast_votek(embeddings=actions_embedding, select_num=3, k=10)
        # random_nn = select_indices[0]
        # randomly select from knns
        random_nn = random.choice(nns[0])

        random_ori = ori_data[random_num]
        random_triple = conv_pool[random_nn]

        # print(random_ori)
        # print("#### votek ####")
        # print(conv_pool[select_indices[0]])
        # print(conv_pool[select_indices[1]])
        # print(conv_pool[select_indices[2]])
        # print("### knn ######")
        # print(conv_pool[nns[0][1]])
        # print(conv_pool[nns[0][2]])
        # print(conv_pool[nns[0][3]])
        # input()

        ori_speakers = recognize_speaker(random_ori['source'].split("</s>")[1].split("\n"))
        triple_speakers = recognize_speaker(random_triple['dialogue'])

        triples_new = []
        for triple in random_triple['triples']:
            tri = triple[0] + " " + triple[1] + " " + triple[2]
            try:
                for si in range(min(len(triple_speakers), len(ori_speakers))):
                    tri = tri.replace(triple_speakers[si], ori_speakers[si])
                triples_new.append(tri)
            except:
                print(triple_speakers)
                print(ori_speakers)
                print(tri)
                print("!!speaker index error!!")
                input()

        new_sample = {"source": " | ".join(triples_new) + " </s> " +  random_ori['source'].split("</s>")[1], "target": "tobefilled"}
        new_samples.append(new_sample)

    for item in new_samples:
        with open("new_conversations.json", 'a+') as f:
            line = json.dumps(item)
            f.write(line+"\n")




if __name__ == "__main__":
    main()