import torch
import torch.nn as nn
import torch.nn.functional as F


class Get_con_score(nn.Module):

    def __init__(self,head_tail,in_dims,out_dims,c):
        super(Get_con_score, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.c = c
        self.linear = {}
        self.linear1 = {}
        self.linear3 = {}
        self.linear_2 = {}
        self.con = {}

        for h in head_tail:
            self.linear[h] = nn.Linear(self.in_dims, self.out_dims)
            gpu1 = torch.device("cuda")
            self.linear[h].to(gpu1)

            self.linear1[h] = nn.Linear(self.out_dims, self.out_dims)
            self.linear1[h].to(gpu1)

            self.linear3[h] = nn.Linear(self.out_dims, self.out_dims)
            self.linear3[h].to(gpu1)

            self.linear_2[h] = nn.Linear(self.out_dims, 1)
            self.linear_2[h].to(gpu1)

            self.con[h] = nn.Parameter(torch.FloatTensor(self.out_dims,self.out_dims))
            nn.init.xavier_uniform_(self.con[h])
            self.register_parameter(h, self.con[h])


    def add(self,adj,node_embedd):
        ##add
        b = torch.mm(adj.t(), node_embedd)  # one hyper embedd for one line

        # ##max
        # adj_t = adj.t().unsqueeze(-1)#w,n,1
        # adj_3 = adj_t.expand(adj_t.shape[0],adj_t.shape[1],self.out_dims)#w,n,512
        #
        # #node_3 = node_embedd.unsqueeze(0).expand(adj_t.shape[0],adj_t.shape[1],self.out_dims)#1,n,512
        # b = adj_3 * node_embedd
        return b


    def forward(self,node_embedding,adj,t):

        #use a mlp
        node_embedd = torch.relu(self.linear[t](node_embedding))
        #use two mlp
        #node_embedd = torch.relu(self.linear1[t](self.linear[t](node_embedding)))
        #use three mlp
        #node_embedd = torch.relu(self.linear3[t](self.linear1[t](self.linear[t](node_embedding))))


        hyper_embedd = self.add(adj,node_embedd)
        #使用w
        #s1_embedd = torch.mm(hyper_embedd, self.con[t])
        #不适用w
        #s1_embedd = hyper_embedd
        #只使用头
        if t == "head":
            s1_embedd = torch.mm(hyper_embedd, self.con[t])
        else:
            s1_embedd = hyper_embedd

        s1_value = torch.sigmoid(self.linear_2[t](s1_embedd))

        return s1_embedd,s1_value

def Jaccard_sim(t_a,t_b):
    a = torch.mul(t_a, t_b)
    a0 = torch.sum(a,dim=1)
    a1 = torch.norm(t_a,p=2,dim=1)
    a2 = torch.norm(t_b,p=2,dim=1)
    a_min = a1+a2+a0+1e-5
    b = torch.div(a0,a_min)

    return b

class Get_rel_score(nn.Module):

    def __init__(self,relation,out_dims2, out_dims,rel_mean):
        super(Get_rel_score, self).__init__()
        self.out_dims2 = out_dims2
        self.out_dims = out_dims
        self.cos = nn.CosineSimilarity(dim=1)
        self.mats = {}

        for mode in relation:#medical 32;iaf 468;uspto 512; IJO 768
            self.mats[relation[mode]] = nn.Parameter(torch.FloatTensor(self.out_dims,self.out_dims2))
            nn.init.xavier_uniform_(self.mats[relation[mode]],rel_mean)
            self.register_parameter(relation[mode]+"_mat", self.mats[relation[mode]])

    def forward(self,head_embedds,tail_embedds):
        s2 = []
        for id,k in enumerate(self.mats.keys()):

            head = torch.mm(head_embedds,self.mats[k])
            tail = torch.mm(tail_embedds,self.mats[k])

            s = self.cos(head,tail)
            #s = Jaccard_sim(head,tail)
            s2.append(s)

        s2 = torch.cat(s2,0).view(len(self.mats),-1)

        return s2.t()


class Get_score(nn.Module):

    def __init__(self,con_model,rel_model,ccuda):
        super(Get_score, self).__init__()
        self.con_model = con_model
        self.rel_model = rel_model
        self.cos = nn.CosineSimilarity(dim=1)
        self.ccuda =ccuda


    def forward(self,data):
        #data
        #data[0] head_node_embedding
        #data[1] adj_head
        #data[2] tail_node_embedding
        #data[3] adj_tail
        #data[4] rel
        #data[5] rel_type_list

        # #head
        head_embedds,head_s1 = self.con_model.forward(data[0],data[1],"head")
        # # #tail
        tail_embedds,tail_s1 = self.con_model.forward(data[2],data[3],"tail")
        # # #rel
        s2 = self.rel_model.forward(head_embedds,tail_embedds)

        #without rel
        # s = self.cos(head_embedds,tail_embedds)# gt=1
        # s_0 = torch.ones_like(s) - s
        # s2 =torch.cat([s_0.unsqueeze(1), s.unsqueeze(1)],1)



#head is one tail are others
        one_tensor = torch.ones(s2.shape[0],s2.shape[1]).cuda()
        score = tail_s1 + s2 - one_tensor
        return s2
        #return score


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=4, size_average=True):
        """
        focal_loss, -α(1-yi)**γ *ce_loss(xi,yi)
        :param alpha:
        :param gamma:
        :param num_classes:
        :param size_average:
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss
        :param preds:    size:[B,N,C] or [B,C]
        :param labels:   size:[B,N] or [B]
        :return:
        """

        preds_softmax = F.softmax(preds,dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
