import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from misc.share_Linear import share_Linear

class _netW(nn.Module):
    def __init__(self, ntoken, ninp, dropout, pretrained_wemb):
        super(_netW, self).__init__()
        #self.word_embed = nn.Embedding(ntoken+1, ninp).cuda()
        self.word_embed = nn.Embedding(ntoken+1, ninp, padding_idx=0).cuda()
        #pdb.set_trace()
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_wemb))
        self.Linear = share_Linear(self.word_embed.weight).cuda()
        #self.init_weights()
        self.d = dropout
    """
    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)
    """

    def forward(self, input, format ='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            out = F.dropout(self.word_embed(input), self.d, training=self.training)

        return out

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())



class _netD(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer. answer_encoder
    """
    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout):
        super(_netD, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, hidden, vocab_size):

        output, _ = self.rnn(input_feat, hidden)
        mask = idx.data.eq(0)  # generate the mask
        mask[idx.data == vocab_size] = 1 # also set the last token to be 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile)

        # Doing self attention here.
        #pdb.set_trace()
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d, training=self.training)).view(idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t(), dim=-1).view(-1,1,idx.size(0))
        feat = torch.bmm(weight, output.transpose(0,1)).view(-1,self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class _netD2(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer. answer_encoder
    """
    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout, fc_dropout):
        super(_netD2, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout
        self.f_d = fc_dropout

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, hidden, vocab_size, encoder_output):

        output, _ = self.rnn(input_feat, hidden)
        mask = idx.data.eq(0)  # generate the mask
        mask[idx.data == vocab_size] = 1 # also set the last token to be 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile)

        # Doing self attention here.
        #pdb.set_trace()
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.f_d, training=self.training)).view(idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t(), dim=-1).view(-1,1,idx.size(0))
        feat = torch.bmm(weight, output.transpose(0,1)).view(-1,self.nhid)
        feat = F.dropout(feat, self.f_d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        batch_size = encoder_output.size(0)
        transform_output = transform_output.view(batch_size, -1, self.ninp)
        encoder_output = encoder_output.view(-1, self.ninp, 1)
        logit = torch.bmm(transform_output, encoder_output)
        logit = logit.view(-1, 100)

        soft_output = F.softmax(logit, dim=-1)
        log_soft_output = F.log_softmax(logit, dim=-1)

        return logit, soft_output, log_soft_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class  LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)

        mask = target.data.gt(0)  # generate the mask
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)
        
        out = torch.masked_select(logprob_select, mask)

        loss = -torch.sum(out) # get the average loss.
        return loss


class nPairLoss(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin, beta):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)
        self.beta = beta

    def forward(self, feat, right, wrong, batch_wrong):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)

        wrong_score = torch.sum(torch.exp((wrong_dis - right_dis.expand_as(wrong_dis)) / self.beta),1) \
                + torch.sum(torch.exp((batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)) / self.beta),1)

        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = right.norm() + feat.norm() + wrong.norm() + batch_wrong.norm()


        loss = (loss_dis + 0.1 * loss_norm) / batch_size

        return loss

class SoftTarget(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    """
    def __init__(self, T=1):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        b = out_t.size(0)
        #pdb.set_trace()
        target = Variable(torch.FloatTensor(b, 100), requires_grad=False).cuda().data.copy_(out_t.data)
        target = Variable(target)
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=-1), F.softmax(target/self.T, dim=-1))*self.T*self.T
        #pdb.set_trace()

        return loss