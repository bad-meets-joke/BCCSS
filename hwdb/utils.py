import os
import struct
import re
from collections import deque

import numpy as np
import scipy.misc
import skimage.exposure
import time
import pdb
import h5py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pdb
import cv2
import random
import time
import copy


def read_gnt_in_directory(gnt_dirpath):
    def samples(f):
        header_size = 10

        # read samples from f until no bytes remaining
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break

            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            assert header_size + width*height == sample_size

            bitmap = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield bitmap, tagcode

    for file_name in os.listdir(gnt_dirpath):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dirpath, file_name)
            with open(file_path, 'rb') as f:
                for bitmap, tagcode in samples(f):
                    yield bitmap, tagcode


def normalize_bitmap(bitmap):
    # pad the bitmap to make it squared
    pad_size = abs(bitmap.shape[0]-bitmap.shape[1]) // 2
    if bitmap.shape[0] < bitmap.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    bitmap = np.lib.pad(bitmap, pad_dims, mode='constant', constant_values=255)

    # rescale and add empty border
    bitmap = scipy.misc.imresize(bitmap, (64 - 4*2, 64 - 4*2))
    bitmap = np.lib.pad(bitmap, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert bitmap.shape == (64, 64)

    bitmap = np.expand_dims(bitmap, axis=0)
    assert bitmap.shape == (1, 64, 64)
    return bitmap


def preprocess_bitmap(bitmap):
    # contrast stretching
    p2, p98 = np.percentile(bitmap, (2, 98))
    assert abs(p2-p98) > 10
    bitmap = skimage.exposure.rescale_intensity(bitmap, in_range=(p2, p98))

    # from skimage.filters import threshold_otsu
    # thresh = threshold_otsu(bitmap)
    # bitmap = bitmap > thresh
    return bitmap


def tagcode_to_unicode(tagcode):
    return struct.pack('>H', tagcode).decode('gb2312')


def unicode_to_tagcode(tagcode_unicode):
    return struct.unpack('>H', tagcode_unicode.encode('gb2312'))[0]


def get_coding_character(coding='gb2312'):
    if coding == 'gb2312':
        start = [0xB0A1]
        end = [0xF7FF]
    elif coding == 'gbk':
        start = [0x8140]
        end = [0xFEFF]
        
    allowed_set = []
    for s, e in zip(start, end):
        for code in range(s, e):
        # 十六进制数转换成字符串，如0xd640转换成'd640'
            hex_str = '%x' % code
            # 从十六进制数字字符串创建一个字节对象
            # 如'd640'转换成b'\xd6@'，其中'@'为ASCII码0x40对应的字符
            bytes_obj = bytes.fromhex(hex_str)
            try:
                allowed_set.append(bytes_obj.decode(coding))
            except Exception as e:
                pass
    return allowed_set


def print_rad(start, end):
    #'wb' must be set, or f.write(str) will report error
    with open('radical_set.txt', 'ab+') as f:
        loc_start = start
        ct = 0
        while loc_start <= end:
            try:
                tmpstr = hex(loc_start)[2:]
                od = (4 - len(tmpstr)) * '0' + tmpstr # 前补0
                ustr = chr(loc_start) #
                index = loc_start - start + 1
                line = (str(index) + '\t' + '0x' + od + '\t' + ustr + '\r\n').encode('utf-8')
                f.write(line)
                loc_start = loc_start + 1
            except Exception as e:
                traceback.print_exc()
                loc_start += 1
                print(loc_start)


def write_rad(): 
    start = [0x2E80, 0x2F00, 0xE815] #, 0xE400] # 部首扩展，康熙部首, PUA部件, 部件扩展
    end = [0x2EF3, 0x2FD5, 0xE864]#, 0xE5E8] # 115，214，81, 452
    for s, e in zip(start, end):
        print_rad(s, e)