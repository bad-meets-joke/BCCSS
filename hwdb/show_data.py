#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from shutil import copy
if not os.path.exists('runs'):
    os.mkdir('runs')


path = os.listdir(os.getcwd())
for p in path:
    if os.path.isdir(p) and p[0] != '_' and p[0] != 'r':
        # print(p)
        for log in os.listdir(p):
            logs = os.path.join(p,log)

            # print (logs)
        print ('cp -r ' + p + '/2019*'  + ' runs/')
        os.system('cp -r ' + p + '/2019*'  + ' runs/')

os.system('python3 -m tensorboard.main --logdir runs')