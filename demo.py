# -*- coding: utf-8 -*-
# @Time    : 19-6-25 上午10:28
# @Author  : Sloan
# @Email   : 630298149@qq.com
# @File    : demo.py
# @Software: PyCharm
import os
import argparse
import logging as log
import time
from statistics import mean
import numpy as np
import torch
from torchvision import transforms as tf
from pprint import pformat

import sys
sys.path.insert(0, '.')

import brambox.boxes as bbb
import vedanet as vn
from utils.envs import initEnv
from vedanet import data as vn_data
from vedanet import models
from utils.test.fast_rcnn.nms_wrapper import nms, soft_nms
import cv2
from PIL import Image

def genResults(reorg_dets, nms_thresh=0.45):
    ret = {}
    for label, pieces in reorg_dets.items():

        for name in pieces.keys():
            pred = np.array(pieces[name], dtype=np.float32)
            keep = nms(pred, nms_thresh, force_cpu=True)
            #keep = soft_nms(pred, sigma=0.5, Nt=0.3, method=1)
            #print k, len(keep), len(pred_dets[k])
            for ik in keep:
                #print k, pred_left[ik][-1], ' '.join([str(int(num)) for num in pred_left[ik][:4]])
                #line ='%f %s' % (pred[ik][-1], ' '.join([str(num) for num in pred[ik][:4]]))
                line = [pred[ik][-1],pred[ik][:4]]
                if label not in ret.keys():
                    ret[label] = []
                ret[label].append(line)

    return ret
def reorgDetection(dets, netw, neth): #, prefix):
    reorg_dets = {}
    for k, v in dets.items():
        #img_fp = '%s/%s.jpg' % (prefix, k)
        img_fp = k #'%s/%s.jpg' % (prefix, k)
        #name = k.split('/')[-1]
        name = k.split('/')[-1][:-4]

        with Image.open(img_fp) as fd:
            orig_width, orig_height = fd.size
        scale = min(float(netw)/orig_width, float(neth)/orig_height)
        new_width = orig_width * scale
        new_height = orig_height * scale
        pad_w = (netw - new_width) / 2.0
        pad_h = (neth - new_height) / 2.0

        for iv in v:
            xmin = iv.x_top_left
            ymin = iv.y_top_left
            xmax = xmin + iv.width
            ymax = ymin + iv.height
            conf = iv.confidence
            class_label = iv.class_label
            #print(xmin, ymin, xmax, ymax)

            xmin = max(0, float(xmin - pad_w)/scale)
            xmax = min(orig_width - 1,float(xmax - pad_w)/scale)
            ymin = max(0, float(ymin - pad_h)/scale)
            ymax = min(orig_height - 1, float(ymax - pad_h)/scale)

            reorg_dets.setdefault(class_label, {})
            reorg_dets[class_label].setdefault(name, [])
            #line = '%s %f %f %f %f %f' % (name, conf, xmin, ymin, xmax, ymax)
            piece = (xmin, ymin, xmax, ymax, conf)
            reorg_dets[class_label][name].append(piece)

    return reorg_dets

def mytest(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    nms_thresh = hyper_params.nms_thresh
    # prefix = hyper_params.prefix
    results = hyper_params.results

    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size, 'labels': labels}
    net = models.__dict__[model_name](hyper_params.classes, weights, train_flag=2, test_args=test_args)
    net.eval()
    log.info('Net structure\n%s' % net)
    #import pdb
    #pdb.set_trace()
    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')

    lb = vn_data.transform.Letterbox(network_size)
    it = tf.ToTensor()
    img_tf = vn_data.transform.Compose([lb, it])
    img_name = 'data/eagle.jpg'
    img = Image.open(img_name)
    img = img_tf(img)
    img = img.unsqueeze(0)
    det = {}
    if use_cuda:
        img = img.cuda()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        output = net(img)
        torch.cuda.synchronize()
        end1_time = time.time()
        det[img_name] = output[0][0]
        #print(det)
    netw, neth = network_size
    reorg_dets = reorgDetection(det, netw, neth)
    result = genResults(reorg_dets, nms_thresh)
    src = cv2.imread(img_name)
    resize = 1
    thresh = 0.3
    print(result)
    for label in result.keys():
        for value in result[label]:
            score = value[0]
            if score>thresh:
                bbox = np.int0(value[1])
                cv2.rectangle(src,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,255,0),4,2)
                cv2.putText(src,'%s %.2f'%(label,score),(bbox[0], bbox[1]-2), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,0), 3,1)
    end2_time = time.time()
    print("net:",end1_time-start_time)
    print("net+draw:",end2_time-start_time)
    dst = cv2.resize(src,(src.shape[1]//resize,src.shape[0]//resize),cv2.INTER_CUBIC)
    cv2.imshow("dst",dst)
    cv2.waitKey(0)
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
    # parser.add_argument('model_name', help='model name', default='Yolov3')
    # args = parser.parse_args()

    train_flag = 2
    config = initEnv(train_flag=train_flag, model_name='Yolov3')

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)

    # init and run eng
    mytest(hyper_params)