
import os, pickle, numpy as np, scipy.sparse as sp, torch

from torch.nn.init import xavier_normal_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm





class Hypergraph(Dataset):
    """
    Loads the hypergraph dataset with train-test split batches
    Parameters:
    args: argparse arguments

    Returns:
    Dictionary with the following key-value pairs:
    1) X: torch tensor of vertex features
    2) train: list of train hyperlinks
    3) test: list of test hyperlinks
    """

    def __init__(self, split):
        path = os.path.join(os.getcwd())
        p = os.path.join(os.getcwd(), '1')

        with open(os.path.join(p, 'indices.pkl'), 'rb') as f:
            D = pickle.load(f)
        # print("D:",D)
        # print(type(D))
        n, m = D['n'], D['m']  # # n:1668 m:2084
        # print("n:",n)
        # print("m:",m)

        if not os.path.isfile(os.path.join(path, "X.npz")):
            X = xavier_normal_(torch.zeros(n, args.d))
        else:
            X = np.load(os.path.join(path, "X.npz"))
            #X = normalise(sp.load_npz(os.path.join(path, "X.npz"))).todense()
            #X = sp.load_npz(os.path.join(path, "X.npz"))).todense()

            args.d = X.shape[1]
        self.X = X
        # print("X:",X.shape) # # （1668，26）
        # print("X:",X)

        # # pos_train   1 1781  2 1739  3 1772
        # # pos_test    1 7170  2 7212  3 7179
        I = D['pos_' + split]  # # (0,562,-1) (0,1258,-1)
        # # #区分正负例的id
        self.pos_id = I
        # print(I)
        iX = np.zeros((len(I), args.d))  # # (正例的个数，特征数)
        for ix, (_, i, _) in enumerate(I):
            # print(ix,i)
            # exit(1)
            iX[ix] = X[i]
        # # neg_train   1 1781  2 1739  3 1772
        # # neg_test    1 7170  2 7212  3 7179
        J = D['neg_' + split]
        # # #区分正负例的id
        self.neg_id = J
        # print(len(J))
        jX = np.zeros((len(J), args.d))
        for jx, (_, j, _) in enumerate(J): jX[jx] = X[j]

        B = []
        if len(I) != len(J): print("The number of positive hyperlinks is not equal to that of negative hyperlinks")
        assert len(I) == len(J)

        for k in range(len(I)):
            b = I[k][0]
            if b != J[k][0]: print("Positive hyperlink ", b, "does not match with negative hyperlink", J[k][0])
            assert b == J[k][0]
            if b == len(B) * args.b: B.append(k)
        B.append(len(I))
        # pos = np.load(os.path.join(p, 'adj_pos_' + str(split) + '.npz'))
        # self.pos_id = list(set(pos['row']))
        # neg = np.load(os.path.join(p, 'adj_pos_' + str(split) + '.npz'))
        # self.neg_id = list(set(neg['row']))
        # print(self.neg_id)
        iA = ssm2tst(
            symnormalise(sp.load_npz(os.path.join(p, 'adj_pos_' + str(split) + '.npz'))))  # # (indices, values, shape

        self.iX = torch.from_numpy(iX).float()
        self.iAX = SparseMM.apply(iA, self.iX)
        # print(self.iAX.shape)
        # print("iX:",self.iX)
        # print(self.iX.shape)
        # print(self.iAX)
        ####################################################HGNN##############################
        import hypergraph_utils as hgut
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        H = None
        tmp = hgut.construct_H_with_KNN(X, K_neigs=[10],
                                        split_diff_scale=False,
                                        is_probH=True, m_prob=1.0)
        H = hgut.hyperedge_concat(H, tmp)
        self.G = torch.Tensor(hgut.generate_G_from_H(H))
        ############################################################################################
        jA = ssm2tst(
            symnormalise(sp.load_npz(os.path.join(p, 'adj_neg_' + str(split) + '.npz'))))  # # (indices, values, shape)
        self.jX = torch.from_numpy(jX).float()
        self.jAX = SparseMM.apply(jA, self.jX)

        BI, BJ = torch.zeros(len(I), args.b), torch.zeros(len(J), args.b)
        for k in range(len(I)):
            b = I[k][0] % args.b
            BI[k][b] = 1 if args.Type == "u" else I[k][2]
            b = J[k][0] % args.b
            BJ[k][b] = 1 if args.Type == "u" else J[k][2]
        self.end = -1
        if b + 1 < args.b: self.end = b + 1

        flag = torch.all(torch.eq(BI, BJ))
        if not flag: print("Indices do not match")
        assert flag
        # print('B:',B)
        # print('I:',BI)
        self.B, self.I = B, BI
        args.n, args.m = n, m
        self.b = args.b

    def __getitem__(self, i):
        s, e = self.B[i], self.B[i + 1]
        Ise = self.I[s:e]
        if i == len(self.B) - 2 and self.end > 0: Ise = self.I[s:e, :self.end]
        # print(self.iAX.shape)
        # print(self.iX.shape)
        # print(self.jAX.shape)
        return {
            "I": Ise,  # # (,64)
            "iX": self.iX[s:e],  # # 正（超边个数，顶点特征个数）
            "jX": self.jX[s:e],  # # 负（超边个数，顶点特征个数）
            "iAX": self.iAX[s:e],  # # 正（顶点个数，顶点特征数 26）
            "jAX": self.jAX[s:e],  # # 负（顶点个数，顶点特证数 26）
            "X": self.X,
            'pos_id': self.pos_id[s:e],
            'neg_id': self.neg_id[s:e],
            'G': self.G
        }

    def __len__(self):
        return len(self.B) - 1




train = DataLoader(Hypergraph('train'), batch_size=1, shuffle=False)
test = DataLoader(Hypergraph('test'), batch_size=1, shuffle=False)