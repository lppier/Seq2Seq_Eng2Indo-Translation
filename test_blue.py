from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
from nltk.translate.bleu_score import sentence_bleu
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

# Loss function: https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

import pandas as pd
from gensim.corpora.dictionary import Dictionary
from nltk import word_tokenize

indo_vocab = {1:'nest',2:'pytorch',3:'length',4:'I'}
reference_tensor = torch.Tensor([1,2,3])
candidate_tensor = torch.Tensor([1,2,3])

def calculate_bleu_score(reference_tensor,candidate_tensor):
    # gram_1_score = sentence_bleu(list(reference_tensor.numpy()),candidate_tensor.numpy(),weights=(1, 0, 0, 0))
    # gram_2_score = sentence_bleu(list(reference_tensor.numpy()),candidate_tensor.numpy(),weights=(0, 1, 0, 0))
    # gram_3_score = sentence_bleu(list(reference_tensor.numpy()),candidate_tensor.numpy(),weights=(0, 0, 1, 0))
    # gram_4_score = sentence_bleu(list(reference_tensor.numpy()),candidate_tensor.numpy(),weights=(0, 0, 0, 1))
    
    # blue_score = (gram_1_score+gram_2_score+gram_3_score+gram_4_score)/4
    # print([list(reference_tensor.numpy())])
    # print(list(candidate_tensor.numpy()))
    # x = [(reference_tensor.numpy().astype(str))]
    # for d in x:
    # 	print(type(d[0]))
    # blue_score = sentence_bleu([(reference_tensor.numpy().astype(str))],candidate_tensor.numpy().astype(str))
    # print("Bleu score",blue_score)

    reference_words = []
    candidate_words = []
    topv, topi = reference_tensor.topk(1)
        #Bleu score

    reference_words.append(indo_vocab.id2token[topi.item()])
    for index in candidate_tensor.data.topk(1):
        candidate_words.append(indo_vocab.id2token[index.item()]) 

    #print("decoder_output:",decoder_output)
    print("reference words",reference_words)
    #print("target_tensor[di]:",target_tensor[di])
    print("candidate words",candidate_words)
    gram_1_score = sentence_bleu(list(reference_words),candidate_words,weights=(1, 0, 0, 0))
    gram_2_score = sentence_bleu(list(reference_words),candidate_words,weights=(0, 1, 0, 0))
    gram_3_score = sentence_bleu(list(reference_words),candidate_words,weights=(0, 0, 1, 0))
    gram_4_score = sentence_bleu(list(reference_words),candidate_words,weights=(0, 0, 0, 1))
    blue_score = (gram_1_score+gram_2_score+gram_3_score+gram_4_score)/4
    print(blue_score)
calculate_bleu_score(reference_tensor,candidate_tensor)    