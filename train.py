
from __future__ import unicode_literals, print_function, division
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from sklearn import model_selection
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.feature_extraction.text import CountVectorizer
import os
from glob import glob
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np
from img2vec_pytorch import Img2Vec
import time

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import pickle
import sys
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet18(pretrained=True)
layer = resnet_model._modules.get('avgpool')
resnet_model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img2vec = Img2Vec(cuda=True)
to_tensor = transforms.ToTensor()






class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = input.view(1,1, 512)
        #print(hidden.size())
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


SOS_token = 0
EOS_token = 1


# class Lang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and EOS

#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)

#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1
# # Turn a Unicode string to plain ASCII, thanks to
# # https://stackoverflow.com/a/518232/2809427
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )

# # Lowercase, trim, and remove non-letter characters


# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s


# def readLangs(lang1, lang2, reverse=False):
#     print("Reading lines...")

#     # Read the file and split into lines
#     lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
#         read().strip().split('\n')

#     # Split every line into pairs and normalize
#     pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

#     # Reverse pairs, make Lang instances
#     if reverse:
#         pairs = [list(reversed(p)) for p in pairs]
#         input_lang = Lang(lang2)
#         output_lang = Lang(lang1)
#     else:
#         input_lang = Lang(lang1)
#         output_lang = Lang(lang2)

#     return input_lang, output_lang, pairs

MAX_LENGTH = 25

# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )


# def filterPair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH and \
#         p[1].startswith(eng_prefixes)


# def filterPairs(pairs):
#     return [pair for pair in pairs if filterPair(pair)]

# def prepareData(lang1, lang2, reverse=False):
#     input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
#     print("Read %s sentence pairs" % len(pairs))
#     pairs = filterPairs(pairs)
#     print("Trimmed to %s sentence pairs" % len(pairs))
#     print("Counting words...")
#     for pair in pairs:
#         input_lang.addSentence(pair[0])
#         output_lang.addSentence(pair[1])
#     print("Counted words:")
#     print(input_lang.name, input_lang.n_words)
#     print(output_lang.name, output_lang.n_words)
#     return input_lang, output_lang, pairs


# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))

# def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]


# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# def tensorsFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        #print(encoder_output[0,0])
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            #print(decoder_input.item())
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    print("Data---------")
    # print(training_pairs[0])
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):
        #training_pair = training_pairs[iter - 1]
        input_paths = joined_df['image_seq'][iter-1]
        sos_img = np.zeros([1,512])
        eos_img = np.ones([1,512])
        input_np=[]
        #print(input_paths)
        for i in input_paths.split(','):
            #print(i)
            img = Image.open(i)
            v = img2vec.get_vec(img,tensor=False).reshape(1,512)
            input_np.append(v)
        
        input_tensor = torch.tensor(input_np).float().to(device)
        #print(input_tensor)
        target_tensor = torch.tensor(joined_df['caption_seq'][iter-1]).to(device)
        #print(target_tensor)
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            torch.save(encoder1.state_dict(),"encoder_model.pkl")
            torch.save(attn_decoder1.state_dict(),"decoder_model.pkl")
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)








crop_path = './detection_data/Images/'

#---------feature extraction------------------
train_flag=sys.argv[1]
if train_flag=='True':

    image_dict = OrderedDict()
    files_list = [y for x in os.walk(crop_path) for y in glob(os.path.join(x[0], '*.jpg'))]
    files_names = [os.path.basename(i) for i in files_list]
    files_names_set = set(files_names)
    print(files_list)

    c=0
    original_file_path = './Images/'
    for i in range(len(files_list)):
        base_name = files_list[i].split("/")[-4]
        b_idx = base_name
        orig_file = os.path.join(original_file_path,base_name)
        image_dict[base_name] = orig_file
        if base_name==files_names[i][:len(base_name)]:
            if base_name not in image_dict.keys():
                image_dict[base_name] = str(files_list[i])
                #image_dict[base_name] += ","+str(orig_file)    
            else:    
                image_dict[b_idx] += ","+str(files_list[i])
        #print(image_dict[b_idx])


    image_dict_keys = [k for k in image_dict.keys()]


    image_df = pd.DataFrame.from_dict(image_dict,orient='index',columns=['image_seq'])
    image_df['image'] = image_df.index
    

    embedding_input_size = len(files_list)
    def first_caption(x):
        return x[0]
    df = pd.read_csv("captions.txt",names=['image','caption'],sep=',',skiprows=1)
    #df = df.groupby("image")['caption'].apply(list).reset_index(name='caption')
    #df['caption']=df['caption'].apply(first_caption)
    print(df.shape)
    print(df.head())
    print(image_df.head())
    joined_df = pd.merge(df,image_df,on='image')
    #joined_df.to_csv("data.csv")

    # print("training pairs")
    # n_iters = 50
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                       for i in range(n_iters)]
    # print(training_pairs[0])

    words = []
    for idx, row in joined_df.iterrows():
        words.extend(joined_df['caption'][idx].lower().split(" "))
        #paragraph context

    words = set(words)
    words = list(words)
    n_words = len(words)+2

    words_idx = {}                                                                                
    words_idx['<SOS>']=0
    words_idx['<EOS>']=1
    for i in range(len(words)):
        words_idx[words[i]]= i+2

    #print(words_idx)

    a_file = open("word_dict.pkl", "wb")
    pickle.dump(words_idx, a_file)


    def tokenizer(x):
        embed = [[0]]
        
        for i in x.lower().split(" "):
          
            embed.append([words_idx[i]])
            
        embed.append([1])
        return embed
    joined_df['caption_seq']=joined_df['caption'].apply(tokenizer)
    def randomizer(x):
        l = x.split(',')

        print('bs',l)
        random.shuffle(l)
        print('as',l)
        return l
    #joined_df['image_seq']=joined_df['image_seq'].apply(randomizer)


    print(joined_df.head())
    print(joined_df.shape)
    print(n_words,embedding_input_size)
    joined_df.to_csv("last_train_data.csv",sep='\t')


    hidden_size = 512
    encoder1 = EncoderRNN(embedding_input_size, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, 39900, print_every=1000)
    torch.save(encoder1.state_dict(),"encoder_model.pkl")
    torch.save(attn_decoder1.state_dict(),"decoder_model.pkl")

else:
    import ast
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method4
    a_file = open("word_dict.pkl", "rb")
    words_idx = pickle.load(a_file)

    max_length=25
    hidden_size = 512
    n_words=8873
    embedding_input_size=42446
    encoder1 = EncoderRNN(embedding_input_size, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, dropout_p=0.1).to(device)
    def eval(jdf,encoder,decoder,i):
        input_paths = jdf['image_seq'][i]
        sos_img = np.zeros([1,512])
        eos_img = np.ones([1,512])
        
        input_np =[]
        #input_paths = ast.literal_eval(input_paths)
        for i in input_paths.split(","):
           
            img = Image.open(i)
            v = img2vec.get_vec(img,tensor=False).reshape(1,512)
            input_np.append(v)
        
        
        input_tensor = torch.tensor(input_np).float().to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        inv = {}
        for key, val in words_idx.items():
            inv[val] = inv.get(val, []) + [key]
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append(['<EOS>'])
                break
            else:
                
                decoded_words.append(inv[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


    ld = pd.read_csv("last_train_data.csv",sep='\t')
    ec = torch.load("encoder_model.pkl")
    encoder1.load_state_dict(ec)
    dc = torch.load("decoder_model.pkl")
    attn_decoder1.load_state_dict(dc)
    #print(joined_df.head(50))
    preds = {}
    refs = {}
    print("Prediction-------------------------------------------------")
    for i in range(87,100):
        print("correct",ld['caption'][i])
        print("prediction-",eval(ld,encoder1,attn_decoder1,i))
        print("-------")

    print("---------------VALIDATION BLEU SCORE----------------------------------- ")
    reference_df = pd.read_csv("captions.txt",names=['image','caption'],sep=',',skiprows=1)
    reference_df = reference_df[39900:]
    reference_df = reference_df.groupby("image")['caption'].apply(list).reset_index(name='caption')
    #print(reference_df.head())
    for i in range(reference_df.shape[0]):
        refs[reference_df['image'][i]]=(reference_df['caption'][i],i)
    
    #print(refs)

    mean_bleu_score = 0
    n=0
    for i in refs.keys():
        true,k = refs[i]
        trues=[]
        for j in true:
            trues.append(j.split(" "))
        
        hypothesis = eval(ld,encoder1,attn_decoder1,k)
        try:
            hypothesis.remove(['<SOS>'])
            hypothesis.remove(['<EOS>'])
        except:
            pass
        preds = []
        for l in hypothesis:
            preds.append(l[0])
       
        mean_bleu_score+=sentence_bleu(trues,preds,smoothing_function=smoothie)
        n+=1

    print("Mean Bleu Score=",mean_bleu_score/n)


    
    










