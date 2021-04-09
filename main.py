import torch
import config
import numpy as np

import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
import torch.nn.functional as F


from data_utils import load_hypers, load_IAF_data
from utils import setup_logging,draw_from_log, draw_from_log_2
from model import Get_con_score,Get_rel_score,Get_score,focal_loss
from torch import optim
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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


log_file = args.log_dir + "/{data:s}-{embed_dim:d}-{lr:f}-{task:s}-{embedding:s}.log".format(

    data=args.data,
    embed_dim=args.embed_dim_out,
    lr=args.lr,
    task = args.task,
    embedding =args.embedding)
logger = setup_logging(log_file)


model_file = args.model_dir + "/{data:s}-{embed_dim:d}-{lr:f}-{task:s}-{embedding:s}.pth".format(
    data=args.data,
    embed_dim=args.embed_dim_out,
    lr=args.lr,
    task = args.task,
    embedding =args.embedding)

print("file",model_file)

picture_file = args.model_dir + "/{data:s}-{embed_dim:d}-{lr:f}-{task:s}-{embedding:s}.jpg".format(
    data=args.data,
    embed_dim=args.embed_dim_out,
    lr=args.lr,
    task = args.task,
    embedding =args.embedding)

print("file",picture_file)

if args.data == "medical":
    train_data, val_data, test_data, weight_ce = load_data_1()
elif args.data == "IAF":
    train_data,test_data,weight_ce = load_data_2()
elif args.data == "rev":
    train_data, test_data, weight_ce = load_data_2()




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
rel_model = Get_rel_score(train_data[4],args.embed_dim_out,args.rel_mean)
model = Get_score(con_model,rel_model,args.cuda)

#train
def run_train():

    if args.cuda:
        model.cuda()

    if args.opt == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0)
    elif args.opt == "adam":

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=args.decay)


    train_losses = []
    val_losses = []
    best_loss = 10000
    best_epoch = 0
    bad_counter = 0

    for i in range(args.epochs):

        optimizer.zero_grad()

        model.train()
        predict = model.forward(train_data)
        train_loss = col_loss(predict,train_data[5],weight_ce)
        train_losses.append(train_loss.item())
        train_loss.backward()


        #输出参数代码 保留
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             if param.grad.mean() == 0:
        #                 flag = True
        #             else:
        #                 flag = False
        #                 print("{}, gradient: {}".format(name, param.grad.mean()))
        #         else:
        #             print("{} has not gradient".format(name))

        optimizer.step()


        if args.data == "medical":
            if i != 0:
                if val_losses[-1] < best_loss:
                    best_loss = val_losses[-1]
                    best_epoch = i
                    bad_counter = 0
                    torch.save(model.state_dict(), model_file)
                    print("save_model_iter", i)
                elif i == args.epochs - 1:
                    torch.save(model.state_dict(), model_file)
                    print("save_model_iter", i)
                else:
                    bad_counter += 1
                    if bad_counter == args.patience:
                        # print("最后一轮，之前所有的loss",val_losses)
                        break
        else:
            if train_losses[-1] < best_loss:
                best_loss = train_losses[-1]
                best_epoch = i
                bad_counter = 0
                torch.save(model.state_dict(), model_file)
                print("save_model_iter",i)
            elif i == args.epochs - 1:
                torch.save(model.state_dict(), model_file)
                print("save_model_iter", i)
            else:
                bad_counter += 1
                if bad_counter == args.patience:
                    break


        if i % args.val_every == 0:
            model.eval()
            with torch.no_grad():

                train_loss,train_acc,train_f1,train_recall,train_precision,train_auc,tran_roc = run_eval(model,train_data)
                test_loss, test_acc, test_f1, test_recall, test_precision, test_auc,test_roc = run_eval(model, test_data)

                logger.info(
                    "test:acu: {};f1_macro: {};recall_macro: {};auc_macro: {};f1_every_class: {};recall_every_class: {};auc_every_class: {}".format(
                        test_acc, test_f1[1], test_recall[1],test_auc[1], test_f1[0],
                        test_recall[0], test_auc[0]))


                logger.info(
                    "train:acu: {};f1_macro: {};recall_macro: {};auc_macro: {};f1_every_class: {};recall_every_class: {};auc_every_class: {}".format(
                        train_acc, train_f1[1], train_recall[1],train_auc[1],  train_f1[0],
                        train_recall[0], train_auc[0]))
                if args.data == "medical":

                    val_loss, val_acc, val_f1, val_recall, val_precision, val_auc,val_roc = run_eval(model, val_data)
                    val_losses.append(val_loss.item())

                    logger.info(
                        "val:acu: {};f1_macro: {};recall_macro: {};auc_macro: {};f1_every_class: {};recall_every_class: {};auc_every_class: {}".format(
                            val_acc, val_f1[1], val_recall[1], val_auc[1], val_f1[0],
                            val_recall[0], val_auc[0]))

                    logger.info(
                        "Iter: {:d};train_loss: {:f};val_loss: {:f};test_loss: {:f};".format(i, train_loss, val_loss, test_loss))
                else:

                    logger.info(
                        "Iter: {:d};train_loss: {:f};test_loss: {:f};".format(i, train_loss,test_loss))



def run_eval(model,data):
    predicts = model.forward(data)
    predicts_list = torch.argmax(F.softmax(predicts, dim=1), dim=1).squeeze().cpu().numpy().tolist()
    golds_list = data[5].cpu()
    loss = col_loss(predicts, data[5], weight_ce)

    acc = accuracy_score(np.array(golds_list), np.array(predicts_list))

    f1_every_class = f1_score(np.array(golds_list), np.array(predicts_list), average=None)
    f1_macro = f1_score(np.array(golds_list), np.array(predicts_list), average="macro")
    f1_micro = f1_score(np.array(golds_list), np.array(predicts_list), average="micro")

    precision_every_class = precision_score(np.array(golds_list), np.array(predicts_list), average=None)
    precision_macro = precision_score(np.array(golds_list), np.array(predicts_list), average="macro")
    precision_micro = precision_score(np.array(golds_list), np.array(predicts_list), average="micro")

    recall_every_class = recall_score(np.array(golds_list), np.array(predicts_list), average=None)
    recall_macro = recall_score(np.array(golds_list), np.array(predicts_list), average="macro")
    recall_micro = recall_score(np.array(golds_list), np.array(predicts_list), average="micro")

    acu_every_class = roc_auc_score(np.eye(len(data[4]))[np.array(golds_list)], predicts.cpu().numpy(),
                                    average=None)
    acu_macro = roc_auc_score(np.eye(len(data[4]))[np.array(golds_list)], predicts.cpu().numpy(),
                              average="macro")
    acu_micro = roc_auc_score(np.eye(len(data[4]))[np.array(golds_list)], predicts.cpu().numpy(),
                              average="micro")

    a = predicts.cpu().numpy()[:,1]
    fpr, tpr, theord = roc_curve(golds_list, np.array(a),pos_label=1)

    f1 = [f1_every_class,f1_macro,f1_micro]
    recall = [recall_every_class,recall_macro,recall_micro]
    precision = [precision_every_class,precision_macro,precision_micro]
    auc = [acu_every_class,acu_macro,acu_micro]
    roc = [fpr, tpr, theord]
    return loss,acc,f1,recall,precision,auc,roc


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
                loss,acc,f1,recall,precision,auc,roc = run_eval(model, test_data)
                print("test:acc: {};f1_macro: {};recall_macro: {};auc_macro: {};f1_every_class: {};auc_every_class: {}".format(acc, f1[1],recall[1], auc[1], f1[0],
                        auc[0]))
                v_loss, v_acc, v_f1, v_recall, v_precision, v_auc,test_roc = run_eval(model, val_data)
                print(
                    "val:acc: {};f1_macro: {};recall_macro: {};auc_macro: {};f1_every_class: {};auc_every_class: {}".format(v_acc, v_f1[1],v_recall[1],v_auc[1],v_f1[0],v_auc[0]))
                plt.title("ROC")
                plt.plot(roc[0], roc[1], 'b')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel('fpr')
                plt.ylabel('recall')
                plt.show()
                f = open(args.model_dir +"roc","a",encoding="utf-8")

                for i in roc:
                    for j in i:
                        f.write(str(j) + "\t")
                    f.write("\n")
                f.close()

                draw_from_log(log_file, picture_file)

            else:
                loss, acc, f1, recall, precision, auc,roc = run_eval(model, test_data)
                print("test:acc: {};f1_macro: {};recall_macro: {};auc_macro: {};f1_every_class: {};auc_every_class: {}".format(
                    acc, f1[1], recall[1], auc[1], f1[0],auc[0]))
                draw_from_log_2(log_file, picture_file)


run_train()

test()







