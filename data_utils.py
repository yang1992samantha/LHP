import pickle
import scipy.sparse as sp

import config
import os
import torch
import numpy as np
from collections import Counter
import torch.nn as nn
args = config.parse()


def read_all_ent(file_name):
    # bu qufen shiti leixing
    ent_id_dict = {}
    f = open(file_name,"r",encoding="utf-8")
    for line in f:
        line = line.strip()
        if line not in ent_id_dict:
            ent_id_dict[line] = len(ent_id_dict)
        else:
            print("重复实体",line)
    f.close()
    return ent_id_dict


def read_embedding(file_name):
    embeddings = np.load(file_name)
    return embeddings


def create_ent_type(ent_id_dict):
    ent_type = {}
    for i in ent_id_dict:
        i_type = i.split("..")[1]
        i_name = i.split("..")[0]
        if i_type not in ent_type:
            ent_type[i_type] = {}
        ent_type[i_type][i_name] = ent_id_dict[i]
    return ent_type


def load_hyper_data(p,ent_id,rel,embedding):
    # get head nodes adj matrix; get head nodes embedding;at last can get head scores list for every hyper;and get head embedding for every hyper
    # get tail nodes adj matrix; get head nodes embedding

    hyper_id = 0
    hypers_head = []
    hypers_tail = []
    rel_type_list = []


    path = os.path.join(args.data, args.dataset, p)
    if args.task == "two_classifier":
        f = open(path,"r",encoding= "utf-8")
        for line in f:
            line = line.strip().split("\t")
            rel_type = line[0]
            if "-1" not in rel_type:
                add_rel_type = 1
                if add_rel_type not in rel:
                    rel[add_rel_type] = "pos"
            elif "-1" in rel_type:
                add_rel_type = 0
                if add_rel_type not in rel:
                    rel[add_rel_type] = "neg"
            # get hyperedge type list
            rel_type_list.append(add_rel_type)
            #####
            # #first node is head and others are tail
            # head_node_id = ent_id[line[1]]
            # hypers_head.append([hyper_id,head_node_id,-1])
            # tail_nodes = line[2:]
            # for n in tail_nodes:
            #     n_id = ent_id[n]
            #     hypers_tail.append([hyper_id,n_id,1])
            # #####

            #the last node is tail and others are head
            tail_node_id = ent_id[line[1]]
            hypers_tail.append([hyper_id,tail_node_id,-1])
            head_nodes = line[2:]
            for n in head_nodes:
                n_id = ent_id[n]
                hypers_head.append([hyper_id,n_id,1])
            ######
            hyper_id = hyper_id + 1
        f.close()
    elif args.task == "multi_classifier":
        f = open(path, "r", encoding="utf-8")
        for line in f:
            new_rel = {v: k for k, v in rel.items()}
            line = line.strip().split("\t")
            rel_type = line[0]

            if "-1" not in rel_type:
                if rel_type not in rel.values():
                    add_rel_type = len(rel) + 1
                    rel[add_rel_type] = rel_type
                else:
                    add_rel_type = new_rel[rel_type]
            elif "-1" in rel_type:
                add_rel_type = 0
                if add_rel_type not in rel:
                    rel[add_rel_type] = "neg"

            rel_type_list.append(add_rel_type)
            # ###
            # # first node is head and others are tail
            # head_node_id = ent_id[line[1]]
            # hypers_head.append([hyper_id, head_node_id, -1])
            # tail_nodes = line[2:]
            # for n in tail_nodes:
            #     n_id = ent_id[n]
            #     hypers_tail.append([hyper_id, n_id, 1])
            # ####
            #the last node is tail and others are head
            tail_node_id = ent_id[line[1]]
            hypers_tail.append([hyper_id,tail_node_id,-1])
            head_nodes = line[2:]
            for n in head_nodes:
                n_id = ent_id[n]
                hypers_head.append([hyper_id,n_id,1])
            ######
            hyper_id = hyper_id + 1
        f.close()

    rel_type_tensor = torch.from_numpy(np.array(rel_type_list))



    #node
    # head nodes embedding
    head_node_embedding = embedding
    # tail nodes embedding
    tail_node_embedding = embedding
    # get head node adj
    adj_head = torch.zeros(len(ent_id), hyper_id)
    adj_tail = torch.zeros(len(ent_id), hyper_id)
    for kh in range(len(hypers_head)):
        node_id = hypers_head[kh][1]#node id
        b = hypers_head[kh][0]#hyperedge id
        adj_head[node_id][b] = 1
    for kt in range(len(hypers_tail)):
        node_id = hypers_tail[kt][1]  # node id
        b = hypers_tail[kt][0]
        adj_tail[node_id][b] = 1
    # head_num = torch.tensor([(h == 1).sum() for h in adj_head.t()])
    # tail_num = torch.tensor([(h == 1).sum() for h in adj_tail.t()])
    # total_num = head_num + tail_num
    # max_node_num = torch.max(total_num)
    # print("----------max_node_num-------------",max_node_num)

    return torch.from_numpy(head_node_embedding).float(),adj_head,torch.from_numpy(tail_node_embedding).float(),adj_tail,rel,rel_type_tensor


def create_weight(rel_type_list):
    rel_type_list = rel_type_list.numpy().tolist()
    weight = []

    rel_id_count = Counter(rel_type_list)
    sort_rel_id = sorted(rel_id_count.items(),key=lambda d:d[0])
    for id,rel in enumerate(sort_rel_id):
        weight.append(len(rel_type_list)/rel[1])

    weight = torch.tensor(np.array(weight), dtype=torch.float32)
    return weight


def load_hypers():

    #load medical data

    ent_id_dict = read_all_ent(os.path.join(args.data, args.dataset, "ent_list"))
    ent_type = create_ent_type(ent_id_dict)

    #load embedding
    if args.embedding == "bert":
        #all_ent_embeddings = read_embedding(os.path.join(args.data, args.dataset, "ent_ch_embedding.npy"))
        all_ent_embeddings = read_embedding(os.path.join(args.data, args.dataset, "bert.npy"))
    elif args.embedding == "gat":
        #all_ent_embeddings = read_embedding(os.path.join(args.data, args.dataset, "encoder_128.npy"))
        all_ent_embeddings = read_embedding(os.path.join(args.data, args.dataset, "gat.npy"))
    elif args.embedding == "word2vector":
        all_ent_embeddings = read_embedding(os.path.join(args.data, args.dataset, "word2vec.npy"))

    elif args.embedding == "random":
        all_ent_embeddings = np.random.uniform(-2,2,(len(ent_id_dict),args.embed_dim_in))#实体个数和256

    rel_type = {}
    train_data = load_hyper_data("train",ent_id_dict,rel_type,all_ent_embeddings)
    val_data = load_hyper_data("val", ent_id_dict, rel_type,all_ent_embeddings)
    test_data = load_hyper_data("test", ent_id_dict, rel_type,all_ent_embeddings)
    weight = create_weight(train_data[5])

    return train_data,val_data,test_data,weight


def normalise(M):
    """
    row-normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)


def create_node_tail_data(data_pos,data_neg,embedding,node_num,hyper_num):

    rel = {0:"neg",1:"pos"}

    #pos 的超边数量
    pos_len = data_pos[-1][0] + 1#13382,53532
    neg_len = data_neg[-1][0] + 1#13382,53532
    if pos_len != neg_len:
        print("error! pos_len is not equal neg_len")
    rel_type =torch.cat([torch.ones(pos_len,dtype=torch.int32),torch.zeros(pos_len,dtype=torch.int32)])
    #负例的每个超边id + pos_len
    new_data_neg = []
    for h in data_neg:
        new_id = h[0] + pos_len
        new_data_neg.append((new_id,h[1],h[2]))

    #正例负例放在一起
    data = data_pos + new_data_neg

    hypers_head = []
    hypers_tail = []
    for h in data:
        if h[-1] == -1:
            hypers_head.append(h)
        elif h[-1] == 1:
            hypers_tail.append(h)
        else:
            print("error")

    head_node_embedding = embedding
    tail_node_embedding = embedding
    ####
    # get head node adj
    adj_head = torch.zeros(node_num, 2*pos_len)
    adj_tail = torch.zeros(node_num, 2*pos_len)
    for kh in range(len(hypers_head)):
        node_id = hypers_head[kh][1]
        b = hypers_head[kh][0]
        adj_head[node_id][b] = 1
    for kt in range(len(hypers_tail)):
        node_id = hypers_tail[kt][1]
        b = hypers_tail[kt][0]
        adj_tail[node_id][b] = 1

    return torch.from_numpy(head_node_embedding).float(), adj_head, torch.from_numpy(
        tail_node_embedding).float(), adj_tail,rel, rel_type


def load_IAF_data():

    #load IAF and REV data

    with open(os.path.join(args.data, args.dataset, 'indices.pkl'), 'rb') as f:
        D = pickle.load(f)
    n, m = D['n'], D['m']#m 超邊:66914 n結點:28798
    #####
    train_pos = D["pos_train"]#1772,86058
    train_neg = D["neg_train"]
    test_pos = D["pos_test"]#7179,338796
    test_neg = D["neg_test"]

    if args.embedding == "random":
        X = nn.init.xavier_normal_(torch.zeros(n,args.embed_dim_in))
        X = X.numpy()
    else:#word2vector
        X = normalise(sp.load_npz(os.path.join(args.data, args.dataset, "X.npz"))).todense()

    #生成train/test的头embedding 头adj
    train_data = create_node_tail_data(train_pos,train_neg,X,n,m)#85864,26764
    test_data = create_node_tail_data(test_pos,test_neg,X,n,m)#40336,11714
    weight = create_weight(train_data[5])#11714

    return train_data, test_data, weight



#load_IAF_data()
#load_hypers()


