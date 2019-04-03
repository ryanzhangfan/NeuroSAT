import argparse
import pickle
import os
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT
from data_maker import generate
import mk_problem

from config import parser

def load_model(args, log_file=None):
  net = NeuroSAT(args)
  net = net.cuda()
  if args.restore:
    if log_file is not None:
      print('restoring from', args.restore, file=log_file, flush=True)
    model = torch.load(args.restore)
    net.load_state_dict(model['state_dict'])
  
  return net

def predict(net, data):
    net.eval()
    outputs = net(data)
    probs = net.vote
    preds = torch.where(outputs>0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
    return preds.cpu().detach().numpy(), probs.cpu().detach().numpy()

if __name__ == '__main__':
  args = parser.parse_args()
  log_file = open(os.path.join(args.log_dir, args.task_name+'.log'), 'a+')
  net = load_model(args)

  TP, TN, FN, FP = 0, 0, 0, 0
  times = []
  for _ in os.listdir(args.data_dir):
    with open(os.path.join(args.data_dir, _), 'rb') as f:
      xs = pickle.load(f)

    for x in xs:
      start_time = time.time()
      preds, probs = predict(net, x)
      end_time   = time.time()
      duration = (end_time - start_time) * 1000
      times.append(duration)

      target = np.array(x.is_sat)
      TP += int(((preds == 1) & (target == 1)).sum())
      TN += int(((preds == 0) & (target == 0)).sum())
      FN += int(((preds == 0) & (target == 1)).sum())
      FP += int(((preds == 1) & (target == 0)).sum())
      
  num_cases = TP + TN + FN + FP
  desc = "%d rnds: tot time %.2f ms for %d cases, avg time: %.2f ms; the pred acc is %.2f, in which TP: %.2f, TN: %.2f, FN: %.2f, FP: %.2f" \
           % (args.n_rounds, sum(times), len(times), sum(times)*1.0/len(times), (TP + TN)*1.0/num_cases, TP*1.0/num_cases, TN*1.0/num_cases, FN*1.0/num_cases, FP*1.0/num_cases)
  print(desc, file=log_file, flush=True)
  
  
'''
for epoch in range(start_epoch, args.epochs):
  if args.train_file is None:
    print('generate data online', file=log_file, flush=True)
    train = generate(args)

  print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc))
  print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=log_file, flush=True)
  print('==> %d/%d epoch, previous best: %.3f' % (epoch+1, args.epochs, best_acc), file=detail_log_file, flush=True)
  train_bar = tqdm(train)
  TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
    desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
    # train_bar.set_description(desc)
    if (_ + 1) % 100 == 0:
      print(desc, file=detail_log_file, flush=True)
  print(desc, file=log_file, flush=True)
  
  val_bar = tqdm(val)
  TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
  for _, prob in enumerate(val_bar):
    optim.zero_grad()
    outputs = net(prob)
    target = torch.Tensor(prob.is_sat).cuda().float()
    # print(outputs.shape, target.shape)
    # print(outputs, target)
    outputs = sigmoid(outputs)
    preds = torch.where(outputs>0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

    TP += (preds.eq(1) & target.eq(1)).cpu().sum()
    TN += (preds.eq(0) & target.eq(0)).cpu().sum()
    FN += (preds.eq(0) & target.eq(1)).cpu().sum()
    FP += (preds.eq(1) & target.eq(0)).cpu().sum()
    TOT = TP + TN + FN + FP
    
    desc = 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP.item()+TN.item())*1.0/TOT.item(), TP.item()*1.0/TOT.item(), TN.item()*1.0/TOT.item(), FN.item()*1.0/TOT.item(), FP.item()*1.0/TOT.item())
    # val_bar.set_description(desc)
    if (_ + 1) % 100 == 0:
      print(desc, file=detail_log_file, flush=True)
  print(desc, file=log_file, flush=True)

  acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
  torch.save({'epoch': epoch+1, 'acc': acc, 'state_dict': net.state_dict()}, os.path.join(args.model_dir, task_name+'_last.pth.tar'))
'''
