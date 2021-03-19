from __future__ import print_function

import argparse
import os
import random
import sys
sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json
import progressbar
import sys
sys.path.append('../')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm

import misc.dataLoader3_train as dl
import misc.model as model
from misc.Img_only import _netE
import datetime
import h5py
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_rcnn_val', default='/data/wanghui/dataset/VisDial_1.0/features_faster_rcnn_x101_val.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_img_rcnn_train', default='/data/wanghui/dataset/VisDial_1.0/features_faster_rcnn_x101_train.h5')
parser.add_argument('--input_ques_h5', default='/data/wanghui/dataset/VisDial_1.0/402020-4/visdial_val_data.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='/data/wanghui/dataset/VisDial_1.0/402020-4/visdial_val_params.json', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output model checkpoints')
parser.add_argument('--encoder', default='Img_only', help='what decoder to use.')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=0, help='number of image split out as validation set.')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.3, help='number of layers')
parser.add_argument('--fc_dropout', type=int, default=0.3, help='number of layers')

opt = parser.parse_args()
#print(opt)

opt.manualSeed = 216#random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
print("Model:", opt.decoder)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' %(t.day, t.month, t.minute)
    save_path = os.path.join(opt.outf, opt.decoder + '.' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass
else:
    # create new folder.
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' %(t.day, t.month, t.minute)
    save_path = os.path.join(opt.outf, opt.decoder + '.' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

####################################################################################
# Data Loader
####################################################################################

dataset = dl.train2(input_img_rcnn=opt.input_img_rcnn_train, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, num_val = opt.num_val, data_split = 'train')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################
vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length + 1
his_length = dataset.ans_length + dataset.ques_length
itow = dataset.itow
img_feat_size = 2048

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, opt.fc_dropout, img_feat_size)
netW = model._netW(vocab_size, opt.ninp, opt.dropout, dataset.pretrained_wemb)
netD = model._netD2(opt.model, opt.ninp, opt.nhid, opt.nlayers, vocab_size, opt.dropout, opt.fc_dropout)


if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])


if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(), netE.cuda(), netD.cuda()

####################################################################################
# training model
soft_labels = np.zeros([123287, 10, 100],dtype='float32')
####################################################################################
def generate_labels():
    netE.eval(), netW.eval(), netD.eval()

    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)
    opt_hidden = netD.init_hidden(opt.batchSize)

    data_iter = iter(dataloader)

    i = 0

    for i in tqdm(range(len(dataloader))):
        
        t1 = time.time()
        data = data_iter.next()
        image, history, question, answer, answerT, answerLen, answerIdx, questionL, \
                                    opt_answerT, opt_answerLen, gt_ids, img_id = data

        batch_size = question.size(0)
        image = image.view(-1, 36, 2048) #   image : batchx36x2048
        img_input.data.resize_(image.size()).copy_(image)

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
            target = gt_ids[:,rnd]
            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            
            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)
            target_input.data.resize_(target.size()).copy_(target)
            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
           
            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            #pdb.set_trace()
            featD, ques_hidden = netE(ques_emb, his_emb, img_input, ques_hidden, hist_hidden, rnd+1)

            opt_emb = netW(opt_ans_input, format='index')
            opt_hidden = repackage_hidden(opt_hidden, opt_emb.size(1))
            logit, soft_output, log_soft_output = netD(opt_emb, opt_ans_input, opt_hidden, vocab_size, featD)

            soft_labels[i][rnd] = logit[0].data

    return soft_labels
####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize, 36, 2048)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)
target_input = torch.LongTensor(opt.batchSize)
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)

if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    opt_ans_input = opt_ans_input.cuda()
    target_input = target_input.cuda()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)
opt_ans_input = Variable(opt_ans_input)
target_input = Variable(target_input)

soft_labels = generate_labels()
f = h5py.File('./Img_only_softlabels.h5','w')
f.create_dataset('softlabels',dtype='float32', data = soft_labels)
f.close()