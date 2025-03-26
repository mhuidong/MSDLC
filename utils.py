import random

import numpy as np
import struct
import torch.nn.functional as F
from collections import Counter

def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

def reorder_data(data, batchsize, iter_num):
    arr = list()
    for i in range(batchsize):
        for j in range(iter_num):
            arr.append(batchsize * j + i)
    return np.array(data[arr])

def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len & 127
        byte_str_len >>= 7
        if byte_str_len == 0:
            f.write(struct.pack('B', this_byte))
            break
        f.write(struct.pack('B', this_byte | 128))
        byte_str_len -= 1

def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
            break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def extend_vocab_size(series, win_len=2, strides=2, final_size=256):
    assert strides <= win_len 
    vocab_size = len(np.unique(series))
    print('The current length of series: {}'.format(len(series)))
    mers = list()
    for start in range(0, len(series)-win_len+1, strides):
        mer = tuple(series[start:start + win_len])
        mers.append(mer)
    n = final_size - vocab_size
    mer_counter = Counter(mers)
    mer_most = mer_counter.most_common(n)
    top_vocabs = [ele[0] for ele in mer_most]

    extend_vocabs = list(range(vocab_size, vocab_size + n))
    extend_dic = {k: v for k, v in zip(top_vocabs, extend_vocabs)} 
    extend2vocab = {k: v for k, v in zip(extend_vocabs, top_vocabs)}
    new_series = [extend_dic.get(ele, ele) for ele in mers] 
    new_series = [item for sublist in new_series for item in (sublist if isinstance(sublist, tuple) else [sublist])]
    print('Extended Length:', len(new_series))
    return np.array(new_series), extend2vocab
