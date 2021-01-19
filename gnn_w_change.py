import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


def gmul(input):
    W, x = input

    x_size = x.size()
    W_size = W.size()
    # print('W_size:',W_size)

    N = W_size[-2]  # N个节点
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3)  # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x)  # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)  # output has size (bs, N, J*num_features)

    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input)  # out has size (bs, N, num_inputs)
        #print(x.size())
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        # print('11111111111111',x.shape)
        x = x.view(-1, self.num_inputs)  # 重新整理x的结构
        x = self.fc(x)  # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class Wcompute(nn.Module):
    def __init__(self,args, input_features, nf, operator='cat', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=True):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.input_features = input_features
        self.c_w_features = args.w_feature_num + args.test_N_way
        self.conv2d_1 = nn.Conv2d(self.c_w_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        self.list = args.w_feature_list
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation
        self.N_way = args.test_N_way
        self.args = args


    def forward(self, x, W_id,adj):
        W1 = x.unsqueeze(2)     # W1 size: bs x N x 1 x num_features
        # if self.N_way == 3:
        #     if self.list == 0:
        #         w_list = [0, 1, 2, 3, 4, 5, 6]  # cli
        #     elif self.list == 1:
        #         w_list = [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # score
        #     elif self.list == 2:
        #         w_list = [0, 1, 2, 16, 17, 18, 19, 20]  # mri
        #     elif self.list == 3:
        #         w_list = list(range(0, 16))  # cli+score
        #     elif self.list == 4:
        #         w_list = [0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20]  # cli+mri
        #     elif self.list == 5:
        #         w_list = [0,1,2]+ list(range(7, 21))  # mri+score
        #     elif self.list == 6:
        #         w_list = list(range(0, 21))  # cli+score+mri
        # else:
        #     if self.list == 0:
        #         w_list = [0, 1, 2, 3, 4, 5]  # cli
        #     elif self.list == 1:
        #         w_list = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # score
        #     elif self.list == 2:
        #         w_list = [0, 1, 2, 15, 16, 17, 18, 19]  # mri
        #     elif self.list == 3:
        #         w_list = list(range(0, 15))  # cli+score
        #     elif self.list == 4:
        #         w_list = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19]  # cli+mri
        #     elif self.list == 5:
        #         w_list = list(range(6, 20))  # mri+score
        #     elif self.list == 6:
        #         w_list = list(range(0, 20))  # cli+score+mri
        w_list = list(range(0,self.args.test_N_way))+list(range(self.args.test_N_way+self.args.clinical_feature_num,self.args.test_N_way+self.args.clinical_feature_num+self.args.w_feature_num))
        W1 = W1[:,:,:, w_list]
        W2 = torch.transpose(W1, 1, 2)  # W2 size: bs x 1 x N x num_feature

        W_new = torch.abs(W1 - W2)  # W_new size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x num_features x N x N
        # W_new = 0-W_new.pow(2)
        # W_new = torch.exp(W_new)

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        if self.args.cuda:
            adj = adj.cuda()
        # W_new = W_new + adj
        W_new = torch.mul(W_new, adj)
        # W_new = W_new + torch.mul(W_new, adj)

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new, dim = 1)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)
        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'cat':
            W_new = torch.cat([W_id, W_new], 3)

        return W_new


class GNN_nl(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J
        # self.num_layers = 2
        self.num_layers = 1

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(args, self.input_features, nf, operator='cat', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(args,self.input_features + int(nf / 2) * i, nf, operator='cat', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(args, self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='cat', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=False)

    def forward(self,  x, adj):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        # print('w_init: ', W_init.shape)

        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init,adj)

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init,adj)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]