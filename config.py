
data = "medical"
dataset = "splits/1"

# data = "IAF"
# dataset = "splits/1"

# data = "uspto"
# dataset = "splits/1"


# data = "iJO1366"
# dataset = "splits/1"

# data = "our-reverb"
# dataset = "splits/1"

'''
gpu: gpu number to use
cuda: True or False
'''
gpu = 0
cuda = True
epochs = 50000


import argparse

def parse():
	"""
	add_arguments and parses arguments
	"""
	p =	argparse.ArgumentParser()
	p.add_argument('--data', type=str, default=data, help='data name (medical)')
	p.add_argument('--dataset', type=str, default=dataset, help='dataset name')
	p.add_argument('--decay', type=float, default=0.0001, help='weight decay')
	p.add_argument('--epochs', type=int, default=epochs, help='number of epochs to train')
	p.add_argument('--gpu', type=int, default=gpu, help='gpu number to use')
	p.add_argument('--cuda', type=bool, default=cuda, help='cuda for gpu')
	p.add_argument('--embed_dim_in', type=int, default=256, help='dim of embedding')#382,256,298,26
	p.add_argument('--embed_dim_out', type=int, default=512, help='dim of embedding')
	p.add_argument('--out_dims2', type=int, default=512, help='dim of embedding')
	p.add_argument('--val_every', type=int, default=1)
	p.add_argument('--opt', type=str, default="adam")
	p.add_argument('--lr', type=float, default=0.001)
	p.add_argument("--log_dir", type=str, default="./")
	p.add_argument("--model_dir", type=str, default="./")
	p.add_argument("--loss", type=str, default="new_focal")
	p.add_argument("--gamma", type=int, default=2)

	p.add_argument("--patience", type=int, default=3000)
	#p.add_argument("--task", type=str, default="multi_classifier")
	p.add_argument("--task", type=str, default="two_classifier")
	p.add_argument("--embedding", type=str, default="word2vector")
	#p.add_argument("--embedding", type=str, default="bert")
	#p.add_argument("--embedding", type=str, default="random")
	p.add_argument("--rel_mean", type=float, default=10)

	return p.parse_args()


