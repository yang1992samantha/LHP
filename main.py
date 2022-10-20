import torch
import config
import numpy as np

from torch.optim import *



import os

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


from data_utils import load_hypers, load_IAF_data
from utils import setup_logging,draw_from_log, draw_from_log_2
from model import Get_con_score,Get_rel_score,Get_score,focal_loss
from torch import optim
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args = config.parse()


def convert_cuda(data):

    new_data0 = data[0].cuda()
    new_data1 = data[1].cuda()
    new_data2 = data[2].cuda()
    new_data3 = data[3].cuda()
    new_data4 = data[4]
    new_data5 = data[5].cuda()
    return (new_data0,new_data1,new_data2,new_data3,new_data4,new_data5)



def load_data_1():
    print("load_data")
    train_data,val_data,test_data,weight = load_hypers()
    if args.cuda:
        train_data = convert_cuda(train_data)
        val_data = convert_cuda(val_data)
        test_data = convert_cuda(test_data)
        weight = weight.cuda()

    return train_data,val_data,test_data,weight

def load_data_2():
    train_data, test_data, weight = load_IAF_data()
    if args.cuda:
        train_data = convert_cuda(train_data)
        test_data = convert_cuda(test_data)
        weight = weight.cuda()
    return train_data,test_data, weight


log_file = args.log_dir + "/{data:s}-{embed_dim:d}-{task:s}-{embedding:s}-{gamma:d}.log".format(

    data=args.data,
    embed_dim=args.embed_dim_out,
    task = args.task,
    embedding =args.embedding,
    gamma=args.gamma)
logger = setup_logging(log_file)

logger.info("-------start-------")
model_file = args.model_dir + "/{data:s}-{embed_dim:d}-{task:s}-{embedding:s}-{gamma:d}.pth".format(
    data=args.data,
    embed_dim=args.embed_dim_out,
    task = args.task,
    embedding =args.embedding,
    gamma=args.gamma)

print("file",model_file)

picture_file = args.model_dir + "/{data:s}-{embed_dim:d}-{lr:f}-{task:s}-{embedding:s}.jpg".format(
    data=args.data,
    embed_dim=args.embed_dim_out,
    lr=args.lr,
    task = args.task,
    embedding =args.embedding)

print("file",picture_file)

if "medical" in args.data:
    train_data, val_data, test_data, weight_ce = load_data_1()
else:
    train_data,test_data,weight_ce = load_data_2()




def col_loss(predict,gold,weight_ce):

    if args.loss == "focal":
        ce = torch.nn.CrossEntropyLoss(weight=None, reduction="mean")
        logp = ce(predict, gold.long())
        p = torch.exp(-logp)
        loss = (args.afa * ((1 - p) ** args.gamma) * logp).mean()
    elif args.loss == "new_focal":
        fl = focal_loss(alpha=weight_ce,gamma=args.gamma,num_classes=4)
        loss = fl.forward(predict,gold.long())

    else:
        ce = nn.CrossEntropyLoss(weight=weight_ce, reduction="mean")
        #ce = nn.CrossEntropyLoss(weight=None, reduction="mean")
        # l = nn.CrossEntropyLoss(weight=weight_CE,reduction="mean")
        loss = ce(predict, gold.long())
    return loss

head_tail = ["head","tail"]
con_model = Get_con_score(head_tail,args.embed_dim_in,args.embed_dim_out,args.cuda)
rel_model = Get_rel_score(train_data[4],args.out_dims2,args.embed_dim_out,args.rel_mean)
model = Get_score(con_model,rel_model,args.cuda)

def run_train():

    if args.cuda:
        model.cuda()

    if args.opt == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0)
    elif args.opt == "adam":

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[40000],gamma = 0.1)

    train_losses = []
    best_auc = 0
    bad_counter = 0

    for i in range(args.epochs):
        model.train()
        predict = model.forward(train_data)
        train_loss = col_loss(predict,train_data[5],weight_ce)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if i % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                if args.data == "medical":
                    val_auc = run_eval(model, val_data)
                else:
                    test_auc = run_eval(model, test_data)

        if args.data == "medical":
            if i != 0:
                if val_auc > best_auc:
                    best_auc = val_auc
                    bad_counter = 0
                    torch.save(model.state_dict(), model_file)
                    logger.info("save_model_iter{:d}".format(i))
                elif i == args.epochs - 1:
                    torch.save(model.state_dict(), model_file)
                    logger.info("save_model_iter{:d}".format(i))
                else:
                    bad_counter += 1
                    if bad_counter == args.patience:
                        # print("最后一轮，之前所有的loss",val_losses)
                        break
        else:
            if test_auc > best_auc:
                best_auc = test_auc
                bad_counter = 0
                torch.save(model.state_dict(), model_file)
                logger.info("save_model_iter{:d}".format(i))
            elif i == args.epochs - 1:
                torch.save(model.state_dict(), model_file)
                logger.info("save_model_iter{:d}".format(i))
            else:
                bad_counter += 1
                if bad_counter == args.patience:
                    break

def run_eval(model,data):
    predicts = model.forward(data)
    predicts_list = torch.argmax(F.softmax(predicts, dim=1), dim=1).squeeze().cpu().numpy().tolist()
    golds_list = data[5].cpu()

    auc_macro = roc_auc_score(np.eye(len(data[4]))[np.array(golds_list)], predicts.cpu().numpy(),
                              average="macro")
    return auc_macro


def test():

    if args.cuda:
        model.cuda()
    if not os.path.isfile(model_file):
        print("no_model_file")
    else:
        model.load_state_dict(torch.load(model_file))
        model.eval()
        with torch.no_grad():
            if args.data == "medical":
                auc = run_eval(model, test_data)
                logger.info("test:auc_macro: {}".format(auc))
            else:
                auc = run_eval(model, test_data)
                logger.info("test:auc_macro: {}".format(auc))

run_train()

test()







