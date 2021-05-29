"""
This repository holds the PyTorch code of our JBHI paper :
*Auto-Metric Graph Neural Network Based on a Meta-learning Strategy
for the Diagnosis of Alzheimer's disease*.

All the materials released in this library can **ONLY** be used for
**RESEARCH** purposes and not for commercial use.

The authors' institution (**Biomedical Image and Health Informatics
Lab,School of Biomedical Engineering, Shanghai Jiao Tong University**)
preserve the copyright and all legal rights of these codes."""

import numpy as np
import torch.utils.data as data
import random
import torch
import gnn_model_w_change as models
import argparse
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from utils import io_utils
import pickle
import os
import time
import textwrap
from pathlib import Path



RESULT_PATH = Path("result/")
DATA_PATH = Path("data/")

parser = argparse.ArgumentParser(description='AMGNN')
parser.add_argument('--metric_network', type=str, default='gnn', metavar='N',
                    help='gnn')
parser.add_argument('--dataset', type=str, default='AD', metavar='N',
                    help='AD')
parser.add_argument('--test_N_way', type=int, default=3, metavar='N')
parser.add_argument('--train_N_way', type=int, default=3, metavar='N')
parser.add_argument('--test_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--train_N_shots', type=int, default=10, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--feature_num', type=int, default=31, metavar='N',
                    help='feature number of one sample')
parser.add_argument('--clinical_feature_num', type=int, default=4, metavar='N',
                    help='clinical feature number of one sample')
parser.add_argument('--w_feature_num', type=int, default=27, metavar='N',
                    help='feature number for w computation')
parser.add_argument('--w_feature_list', type=int, default=5, metavar='N',
                    help='feature list for w computation')
# 0-4,1-9，2-5,3-13,4-9，5-14,6-18
# 0-4,1-9，2-10,3-13,4-14，5-19,6-23
parser.add_argument('--iterations', type=int, default=500, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--dec_lr', type=int, default=10000, metavar='N',
                    help='Decreasing the learning rate every x iterations')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_train', type=int, default=64, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--unlabeled_extra', type=int, default=0, metavar='N',
                    help='Number of shots when training')
parser.add_argument('--test_interval', type=int, default=200, metavar='N',
                    help='how many batches between each test')
parser.add_argument('--random_seed', type=int, default=2019, metavar='N')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('GPU:', args.cuda)
random_seed = args.random_seed
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def setup_seed(seed=random_seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class Generator(data.DataLoader):
    def __init__(self, root,keys = ['CN','MCI', 'AD']):
        with open(root, 'rb') as load_data:
            data_dict = pickle.load(load_data)
        data_ = {}
        for i in range(len(keys)):
            data_[i]= data_dict[keys[i]]
        self.data = data_
        self.channal = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(self, batch_size=5, n_way=4, num_shots=10, unlabeled_extra=0, cuda=False, variable=False):
        # init
        batch_x = np.zeros((batch_size,self.channal,self.feature_shape[0],self.feature_shape[1]), dtype='float32')  # features
        labels_x = np.zeros((batch_size, n_way), dtype='float32')  # labels
        labels_x_global = np.zeros(batch_size, dtype='int64')
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(np.zeros((batch_size,self.channal,self.feature_shape[0],self.feature_shape[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype='float32')))

        # feed data

        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if class_num == pre_class:
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    # Test sample

                    batch_x[batch_counter,0, :,:] = samples[0]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                for samples_num in range(len(samples)):
                    try:
                        batches_xi[indexes_perm[counter]][batch_counter, :] = samples[samples_num]
                    except:
                        print(samples[samples_num])

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    # target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(pre_class)

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)

        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

def compute_adj(batch_x, batches_xi):
    x = torch.squeeze(batch_x)
    xi_s = [torch.squeeze(batch_xi) for batch_xi in batches_xi]

    nodes = [x] + xi_s
    nodes = [node.unsqueeze(1) for node in nodes]
    nodes = torch.cat(nodes, 1)
    age = nodes.narrow(2, 0, 1)
    age = age.cpu().numpy()
    gender = nodes.narrow(2, 1, 1)
    gendre = gender.cpu().numpy()
    apoe = nodes.narrow(2, 2, 1)
    apoe = apoe.cpu().numpy()
    edu = nodes.narrow(2, 3, 1)
    edu = edu.cpu().numpy()
    adj = np.ones(
        (args.batch_size, args.train_N_way * args.train_N_shots + 1, args.train_N_way * args.train_N_shots + 1, 1),
        dtype='float32')+4

    for batch_num in range(args.batch_size):
        for i in range(args.train_N_way * args.train_N_shots + 1):
            for j in range(i + 1, args.train_N_way * args.train_N_shots + 1):
                if np.abs(age[batch_num, i, 0] - age[batch_num, j, 0]) <= 0.06:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if np.abs(edu[batch_num, i, 0] - edu[batch_num, j, 0]) <= 0.14:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if gendre[batch_num, i, 0] == gendre[batch_num, j, 0]:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
                if apoe[batch_num, i, 0] == apoe[batch_num, j, 0]:
                    adj[batch_num, i, j, 0] -= 1
                    adj[batch_num, j, i, 0] -= 1
    adj = 1/adj
    adj = torch.from_numpy(adj)
    return adj.cuda()

def train_batch(model, data):
    [amgnn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi] = data
    z_clinical, z_mri_feature = batch_x[:, 0, 0, 0:args.clinical_feature_num], batch_x[:, :, :, args.clinical_feature_num:]
    zi_s_clinical = [batch_xi[:,0,0,0:args.clinical_feature_num] for batch_xi in batches_xi]
    zi_s_mri_feature = [batch_xi[:, :, :, args.clinical_feature_num:] for batch_xi in batches_xi]

    adj = compute_adj(z_clinical,zi_s_clinical)

    out_metric, out_logits = amgnn(inputs=[z_clinical, z_mri_feature, zi_s_clinical, zi_s_mri_feature, labels_yi, oracles_yi,adj])
    logsoft_prob = softmax_module.forward(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)
    loss.backward()

    return loss


def test_one_shot(args, fold,test_root, model, test_samples=50, partition='test',io_path= 'run.log'):
    io = io_utils.IOStream(io_path)

    io.cprint('\n**** TESTING BEGIN ***' )
    root = test_root
    loader = Generator(root,keys = ['CN','MCI','AD'])
    [amgnn, softmax_module] = model
    amgnn.eval()
    correct = 0
    total = 0
    pre_all = []
    pre_all_num = []
    real_all = []
    iterations = int(test_samples / args.batch_size_test)
    for i in range(iterations):
        data = loader.get_task_batch(batch_size=args.batch_size_test, n_way=args.test_N_way,
                                     num_shots=args.test_N_shots, unlabeled_extra=args.unlabeled_extra,cuda = args.cuda)
        [x_t, labels_x_cpu_t, _, _, xi_s, labels_yi_cpu, oracles_yi] = data

        z_clinical, z_mri_feature = x_t[:, 0, 0, 0:args.clinical_feature_num],x_t[:, :, :,args.clinical_feature_num:]
        zi_s_clinical = [batch_xi[:, 0, 0, 0:args.clinical_feature_num] for batch_xi in xi_s]
        zi_s_mri_feature = [batch_xi[:, :, :,args.clinical_feature_num:] for batch_xi in xi_s]

        adj = compute_adj(z_clinical, zi_s_clinical)

        x = x_t
        labels_x_cpu = labels_x_cpu_t

        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in zi_s_mri_feature]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in zi_s_mri_feature]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
        z_mri_feature = Variable(z_mri_feature)

        # Compute metric from embeddings
        output, out_logits = amgnn(inputs=[z_clinical, z_mri_feature, zi_s_clinical, xi_s, labels_yi, oracles_yi,adj])
        output = out_logits
        Y = softmax_module.forward(output)
        y_pred = softmax_module.forward(output)

        y_pred = y_pred.data.cpu().numpy()
        y_inter = [list(y_i) for y_i in y_pred]
        pre_all_num = pre_all_num + list(y_inter)
        y_pred = np.argmax(y_pred, axis=1)
        labels_x_cpu = labels_x_cpu.cpu().numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
        pre_all = pre_all+list(y_pred)
        real_all = real_all + list(labels_x_cpu)
        for row_i in range(y_pred.shape[0]):
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1
    labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    loss_test_f = float(loss_test)
    del loss_test
    io.cprint('real_label:  '+str(real_all))
    io.cprint('pre_all:  '+str(pre_all))
    io.cprint('pre_all_num:  '+str(pre_all_num))
    io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0 * correct / total))
    io.cprint('*** TEST FINISHED ***\n'.format(correct, total, 100.0 * correct / total))

    amgnn.train()

    return 100.0 * correct / total, loss_test_f


if __name__ =='__main__':
    ################################################################
    print(time.strftime("%F"))

    timedata = time.strftime("%F")
    name = timedata + '-3-classe'
    save_path = RESULT_PATH / name

    if name not in os.listdir(RESULT_PATH):
        os.makedirs(save_path)
    io = io_utils.IOStream(RESULT_PATH / 'run.log')
    print('The result will be saved in :', save_path)
    setup_seed(args.random_seed)

    amgnn = models.create_models(args, cnn_dim1=2)
    io.cprint(str(amgnn))
    softmax_module = models.SoftmaxModule()
    print(amgnn.parameters())
    if args.cuda:
        amgnn.cuda()

    weight_decay = 0

    opt_amgnn = optim.Adam(amgnn.parameters(), lr=args.lr, weight_decay=weight_decay)
    amgnn.train()
    counter = 0
    total_loss = 0
    val_acc, val_acc_aux = 0, 0
    test_acc = 0
    for batch_idx in range(args.iterations):

        root = DATA_PATH / 'AD_3_CLASS_TRAIN.pkl'
        da = Generator(root, keys=['CN', 'MCI','AD'])
        data = da.get_task_batch(batch_size=args.batch_size_train, n_way=args.train_N_way,
                                 num_shots=args.train_N_shots, unlabeled_extra=args.unlabeled_extra, cuda=args.cuda)
        [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi] = data

        opt_amgnn.zero_grad()

        loss_d_metric = train_batch(model=[amgnn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi])
        opt_amgnn.step()

        adjust_learning_rate(optimizers=[opt_amgnn], lr=args.lr, iter=batch_idx)

        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss_d_metric.item()
        if batch_idx % args.log_interval == 0:
            display_str = 'Train Iter: {}'.format(batch_idx)
            display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss / counter)
            io.cprint(display_str)
            counter = 0
            total_loss = 0
        ####################
        # Test
        ####################
        if (batch_idx + 1) % args.log_interval == 0:

            test_samples = 320
            test_root = DATA_PATH / 'AD_3_CLASS_TEST.pkl'
            test_acc_aux, test_loss_ = test_one_shot(args, 0, test_root, model=[amgnn, softmax_module],
                                                     test_samples=test_samples, partition='test',
                                                     io_path=save_path / 'run.log')
            amgnn.train()

            if test_acc_aux is not None and test_acc_aux >= test_acc:
                test_acc = test_acc_aux
                # val_acc = val_acc_aux
                torch.save(amgnn, save_path / 'amgnn_best_model.pkl')
            if args.dataset == 'AD':
                io.cprint("Best test accuracy {:.4f} \n".format(test_acc))

