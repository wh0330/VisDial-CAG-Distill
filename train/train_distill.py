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
from misc.metrics import NDCG
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
                    decode_txt, sample_batch_neg, l2_norm

import misc.dataLoader_train as dl
import misc.model as model
import misc.CAGraph as CAGraph
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_rcnn_val', default='/data/wanghui/dataset/VisDial_1.0/features_faster_rcnn_x101_val.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_img_rcnn_train', default='/data/wanghui/dataset/VisDial_1.0/features_faster_rcnn_x101_train.h5')
parser.add_argument('--input_ques_h5', default='/data/wanghui/dataset/VisDial_1.0/402020-4/visdial_val_data.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='/data/wanghui/dataset/VisDial_1.0/402020-4/visdial_val_params.json', help='path to dataset, now json file')
parser.add_argument('--dense_annotations', default='/data/wanghui/dataset/VisDial_1.0/visdial_1.0_val_dense_annotations.json', help='path to dataset, now json file')
parser.add_argument('--softlabel', default='/data/wanghui/project/VD-CAG-1.0-NDCG/image_only_softlabels.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save', help='folder to output model checkpoints')
parser.add_argument('--encoder', default='CAG_Distill', help='what decoder to use.')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=0, help='number of image split out as validation set.')

parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')
parser.add_argument('--start_epoch', type=int, default=1, help='start of epochs to train for')
parser.add_argument('--teacher_forcing', type=int, default=1, help='start of epochs to train for')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--save_iter', type=int, default=1, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose'  , action='store_true', help='show the sampled caption')

parser.add_argument('--conv_feat_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=512, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.3, help='number of layers')
parser.add_argument('--fc_dropout', type=int, default=0.3, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--log_interval', type=int, default=200, help='how many iterations show the log info')

opt = parser.parse_args()
opt.manualSeed = 216#random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
print("Model:", opt.encoder)
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
    save_path = os.path.join(opt.outf, opt.encoder + '.' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass
else:
    # create new folder.
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' %(t.day, t.month, t.minute)
    save_path = os.path.join(opt.outf, opt.encoder + '.' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass


####################################################################################
# Data Loader
####################################################################################

dataset = dl.train_distill(input_img_rcnn=opt.input_img_rcnn_train, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'train', softlabels=opt.softlabel)

dataset_val = dl.validate(input_val_rcnn=opt.input_img_rcnn_val, input_ques_h5=opt.input_ques_h5,
                input_json=opt.input_json, dense_annotations = opt.dense_annotations, negative_sample = opt.negative_sample,
                num_val = opt.num_val, data_split = 'test')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                         shuffle=False, num_workers=int(opt.workers))

####################################################################################
# Build the Model
####################################################################################
n_neg = opt.negative_sample
vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length + 1
his_length = dataset.ans_length + dataset.ques_length
itow = dataset.itow
img_feat_size = 512

netE = CAGraph._netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout)
netW = model._netW(vocab_size, opt.ninp, opt.dropout, dataset.pretrained_wemb)
netD = model._netD2(opt.model, opt.ninp, opt.nhid, opt.nlayers, vocab_size, opt.dropout, opt.fc_dropout)
criterion = nn.CrossEntropyLoss()
KL_criterion = model.SoftTarget(1)
ndcg = NDCG()

if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])


if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(),  netE.cuda(), 
    netD.cuda(),  criterion.cuda(),  KL_criterion.cuda()

####################################################################################
# training model
####################################################################################
def train(epoch):
    netW.train(), netE.train(), netD.train()

    lr = adjust_learning_rate(optimizer, epoch, opt.lr, 4)

    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)
    opt_hidden = netD.init_hidden(opt.batchSize)


    data_iter = iter(dataloader)

    average_loss = 0
    CE_average_loss = 0
    KL_average_loss = 0
    count = 0
    i = 0

    while i < len(dataloader):
        
        t1 = time.time()
        data = data_iter.next()
        image, history, question, answer, answerT, answerLen, answerIdx, questionL, \
                                    opt_answerT, opt_answerLen, gt_ids, img_id, soft_label= data

        batch_size = question.size(0)
        image = image.view(-1, 36, 2048) #   image : batchx36x2048
        img_input.data.resize_(image.size()).copy_(image)

        for rnd in range(10):
            netW.zero_grad(), netE.zero_grad(), netD.zero_grad()
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()
            target = gt_ids[:,rnd]
            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            soft_target = soft_label[:,rnd]
            
            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)
            target_input.data.resize_(target.size()).copy_(target)
            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
            soft_target_input.data.resize_(soft_target.size()).copy_(soft_target)
           
            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            featD, ques_hidden = netE(ques_emb, his_emb, img_input, ques_hidden, hist_hidden, rnd+1)

            opt_emb = netW(opt_ans_input, format='index')
            opt_hidden = repackage_hidden(opt_hidden, opt_emb.size(1))
            logit, soft_output, log_soft_output = netD(opt_emb, opt_ans_input, opt_hidden, vocab_size, featD)

            CE_loss = criterion(logit, target_input)
            KL_loss = KL_criterion(logit, soft_target_input * 5) * 10
            total_loss = CE_loss + KL_loss
            
            average_loss += total_loss.data[0]
            CE_average_loss += CE_loss.data[0]
            KL_average_loss += KL_loss.data[0]
            total_loss.backward()

            optimizer.step()
            count += 1

        i += 1
        if i % opt.log_interval == 0:
            average_loss /= count
            CE_average_loss /= count
            KL_average_loss /= count
            print("step {} / {} (epoch {}), d_l {:.3f}, c_l {:.3f}, k_l {:.3f}, lr = {:.6f}"\
                .format(i, len(dataloader), epoch, average_loss, CE_average_loss, KL_average_loss, lr))
            average_loss = 0
            CE_average_loss = 0
            KL_average_loss = 0
            count = 0
    return average_loss, lr


def val():
    netE.eval(), netW.eval(), netD.eval()

    n_neg = 100
    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_hidden(opt.batchSize)
    hist_hidden = netE.init_hidden(opt.batchSize)

    opt_hidden = netD.init_hidden(opt.batchSize)
    i = 0

    average_loss = 0
    rank_all_tmp = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id , round_id, gt_relevance = data

        batch_size = question.size(0)

        image = image.view(-1, 36, 2048) #   image : batchx36x2048
        img_input.data.resize_(image.size()).copy_(image)
        round_id = round_id.cuda()

        for rnd in range(10):
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            gt_id = answer_ids[:,rnd]

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)
            relevance.data.resize_(gt_relevance.size()).copy_(gt_relevance)

            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
            gt_index.data.resize_(gt_id.size()).copy_(gt_id)
            opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))


            featD, ques_hidden = netE(ques_emb, his_emb, img_input, ques_hidden, hist_hidden, rnd+1)

            opt_ans_emb = netW(opt_ans_input, format = 'index')
            opt_hidden = repackage_hidden(opt_hidden, opt_ans_input.size(1))
            logit, soft_output, log_soft_output = netD(opt_ans_emb, opt_ans_input, opt_hidden, vocab_size, featD)

            if rnd == (round_id[0]-1):
                ndcg.observe(logit, relevance)

            #pdb.set_trace()

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b*100

            gt_score = logit.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(logit, 1, descending=True)

            count = sort_score.gt(gt_score.view(-1,1).expand_as(sort_score))
            rank = count.sum(1) + 1
            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())
            
        i += 1
        sys.stdout.write('Evaluating: {:d}/{:d}  \r' \
          .format(i, len(dataloader_val)))
        sys.stdout.flush()

    return rank_all_tmp

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize, 36, 2048)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)
target_input = torch.LongTensor(opt.batchSize)
soft_target_input = torch.FloatTensor(opt.batchSize)
relevance = torch.FloatTensor(opt.batchSize, 100)
# answer input
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)


if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    opt_ans_input = opt_ans_input.cuda()
    gt_index = gt_index.cuda()
    relevance = relevance.cuda()
    target_input = target_input.cuda()
    soft_target_input = soft_target_input.cuda()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)
opt_ans_input = Variable(opt_ans_input)
gt_index = Variable(gt_index)
relevance = Variable(relevance)
target_input = Variable(target_input)
soft_target_input = Variable(soft_target_input)

optimizer = optim.Adam([{'params': netW.parameters()},
                        {'params': netE.parameters()},
                        {'params': netD.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))


history = []

for epoch in range(opt.start_epoch,opt.niter):

    t = time.time()

    train_loss, lr = train(epoch)
    print ('Epoch: %d learningRate %4f train loss %4f Time: %3f' % (epoch, lr, train_loss, time.time()-t))
    train_his = {'loss': train_loss}

    print('Evaluating CAG model... ')
    rank_all = val()
    ndcg_dic = ndcg.retrieve(reset=True)
    R1 = np.sum(np.array(rank_all)==1) / float(len(rank_all))
    R5 =  np.sum(np.array(rank_all)<=5) / float(len(rank_all))
    R10 = np.sum(np.array(rank_all)<=10) / float(len(rank_all))
    ave = np.sum(np.array(rank_all)) / float(len(rank_all))
    mrr = np.sum(1/(np.array(rank_all, dtype='float'))) / float(len(rank_all))
    print ('%d/%d: NDCG: %f mrr: %f R1: %f R5 %f R10 %f Mean %f' %(1, len(dataloader_val), ndcg_dic['ndcg'], mrr, R1, R5, R10, ave))
    val_his = {'NDCG':ndcg_dic['ndcg'],'R1': R1, 'R5':R5, 'R10': R10, 'Mean':ave, 'mrr':mrr}
    history.append({'epoch':epoch, 'train': train_his, 'val': val_his})

    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netW': netW.state_dict(),
                    'netD': netD.state_dict(),
                    'netE': netE.state_dict()},
                    '%s/epoch_%d.pth' % (save_path, epoch))

        json.dump(history, open('%s/log.json' %(save_path), 'w'))
