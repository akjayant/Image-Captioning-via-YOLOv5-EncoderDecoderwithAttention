
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



MAX_LENGTH = 25

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
        input_np=[sos_img]
        for i in input_paths.split(','):
            img = Image.open(i)
            v = img2vec.get_vec(img,tensor=False).reshape(1,512)
            input_np.append(v)
        input_np.append(eos_img)
        input_tensor = torch.tensor(input_np).float().to(device)
        #print(input_tensor)
        target_tensor = torch.tensor(joined_df['caption_seq'][iter-1]).to(device)
        #print(target_tensor)
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            torch.save(encoder1.state_dict(),"encoder_model")
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




flag = sys.argv[1]
if flag=True:



	crop_path = '/media/ashish-j/B/wheat_detection/flick_data/detection_data/Images/'

	#---------feature extraction------------------


	image_dict = OrderedDict()
	files_list = [y for x in os.walk(crop_path) for y in glob(os.path.join(x[0], '*.jpg'))]
	files_names = [os.path.basename(i) for i in files_list]
	files_names_set = set(files_names)

	c=0
	for i in range(len(files_list)):
	    base_name = files_list[i].split("/")[-4]
	    #print(base_name)
	    b_idx = base_name
	    if base_name==files_names[i][:len(base_name)]:
	        if base_name not in image_dict.keys():
	            image_dict[b_idx] = str(files_list[i])
	        else:
	            image_dict[b_idx] += ","+str(files_list[i])


	image_dict_keys = [k for k in image_dict.keys()]


	image_df = pd.DataFrame.from_dict(image_dict,orient='index',columns=['image_seq'])
	image_df['image'] = image_df.index
	idx =  [i for i in range(image_df.shape[0])]
	#print(image_df.head())


	embedding_input_size = len(files_list)
	def first_caption(x):
	    return x[0]
	df = pd.read_csv("captions.txt",names=['image','caption'],sep=',')
	df = df.groupby("image")['caption'].apply(list).reset_index(name='caption')
	df['caption']=df['caption'].apply(first_caption)
	#print(df.shape)
	#print(df.head())

	joined_df = pd.merge(df,image_df,on='image')
	#joined_df.to_csv("data.csv")

	# print("training pairs")
	# n_iters = 50
	# training_pairs = [tensorsFromPair(random.choice(pairs))
	#                       for i in range(n_iters)]
	# print(training_pairs[0])

	words = []
	for idx, row in joined_df.iterrows():
	    #title
	    words.extend(joined_df['caption'][idx].split(" "))
	    #paragraph context

	words = set(words)
	words = list(words)
	n_words = len(words)+2
	                                                                                     
	words_idx = {'<SOS>':0,
	'<EOS>':1}
	for i in range(len(words)):
	    words_idx[words[i]]= i+2



	
    a_file = open("word_dict.pkl", "wb")
    pickle. dump(words_idx, a_file)

	def tokenizer(x):
	    embed = []
	    embed.append([0])
	    for i in x.split(" "):
	        try:
	            embed.append([words_idx[i]])
	        except:
	            pass
	    embed.append([1])
	    return embed
	joined_df['caption_seq']=joined_df['caption'].apply(tokenizer)

	print(joined_df.head())
	print(joined_df['caption_seq'][0])



	hidden_size = 512
	encoder1 = EncoderRNN(embedding_input_size, hidden_size).to(device)
	attn_decoder1 = AttnDecoderRNN(hidden_size, n_words, dropout_p=0.1).to(device)
	trainIters(encoder1, attn_decoder1, 7995, print_every=500)
	torch.save(encoder1.state_dict(),"encoder_model")
	torch.save(attn_decoder1.state_dict(),"decoder_model.pkl")
	joined_df.to_csv("last_train_data.csv",sep='\t')


else:
	a_file = open("word_dict.pkl", "rb")
	words_idx = pickle.load(a_file)

	max_length=25
	def eval(jdf,encoder,decoder,i):
	    input_paths = joined_df['image_seq'][i]
	    sos_img = np.zeros([1,512])
	    eos_img = np.ones([1,512])
	    input_np=[sos_img]
	    for i in input_paths.split(','):
	        img = Image.open(i)
	        v = img2vec.get_vec(img,tensor=False).reshape(1,512)
	        input_np.append(v)
	    input_np.append(eos_img)
	    
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
	            decoded_words.append('<EOS>')
	            break
	        else:
	            
	            decoded_words.append(inv[topi.item()])

	        decoder_input = topi.squeeze().detach()

	    return decoded_words


	ld = pd.read_csv("last_train_data.csv",sep='\t')
	ec = torch.load("encoder_model")
	encoder1.load_state_dict(ec)
	dc = torch.load("decoder_model.pkl")
	attn_decoder1.load_state_dict(dc)
	#print(joined_df.head(50))
	print("prediction-------------------------------------------------")
	for i in [1,2,3,4,5,6,7,8,9,10]:
	    print("correct",joined_df['caption'][i])
	    print("prediction-",eval(ld,encoder1,attn_decoder1,i))
	    print("-------")









