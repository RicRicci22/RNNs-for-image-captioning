import numpy as np 
import argparse
import torch
import torch.nn as nn
import math
import pickle 
import os 
from itertools import chain
from tqdm import tqdm


def parameter_parser():

    parser = argparse.ArgumentParser(description = "Text Generation")

    parser.add_argument("--epochs", dest="num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--load_model", dest="load_model", type=bool, default=True)
    parser.add_argument("--dataset", dest="dataset", type=str, default='ucm')
    parser.add_argument("--feature_extractor", dest="feature_extractor", type=str, default='resnet152')
    parser.add_argument("--model_type", dest="model_type", type=str, default='gru')
                        
    return parser.parse_args()


#### WORD LEVEL GENERATION UTILITIES ####
def convert_to_words_ucm(input_file):
    with open(input_file,'r',encoding='utf-8') as file:
        content = file.read().lower()    
    sentences = dict()
    max_l = 0
    for row in content.split('\n'):
        pieces = row.strip().split(' ') 
        pieces.append('endseq')
        pieces.insert(1,'startseq')
        filename = (pieces[0])
        del(pieces[0])
        try:
            sentences[filename].append(pieces)
        except:
            sentences[filename] = []
            sentences[filename].append(pieces)
        
        if(len(pieces))>max_l:
            max_l = len(pieces)

    return sentences,max_l


def word_frequency_ucm_uav(text_in_words,min_word_frequency,test_sentences):
    word_freq = {}
    for question in text_in_words:
        for word in question:
            if(word in word_freq.keys()):
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] <= min_word_frequency:
            ignored_words.add(k)
    print('Unique words before ignoring:', len(word_freq.keys()))
    print('Ignoring words with frequency <', min_word_frequency)
    words = [k for k in word_freq.keys() if k not in ignored_words]
    print('Unique words after ignoring:', len(words))

    word_indices = dict((c, i+1) for i, c in enumerate(words))
    indices_word = dict((i+1, c) for i, c in enumerate(words))
    
    # Add unk token and padding
    word_indices['pad'] = 0
    indices_word[0] = 'pad'
    
    # Add unknown words from test sentences
    for image_sentences in test_sentences:
        for sentence in image_sentences:
            for word in sentence:
                try:
                    a = word_indices[word]
                except:
                    ignored_words.add(word)
    
    return word_indices,indices_word,ignored_words


def create_sequences_ucm(text_dictionary,filenames,ignored_words):
    # cut the text in semi-redundant sequences of SEQUENCE_LEN words
    sentences = list()
    # Create the sentences 
    for i in filenames:
        descriptions = text_dictionary[i]
        for description in descriptions:
            for j in range(0, len(description)-1):
                if not (any(item in ignored_words for item in description[:2+j])):
                    sentences.append((i,description[0:1+j]))

    return sentences
           
def create_lists_ucm_uav(train_filenames,val_filenames,test_filenames):
    train_ = []
    val_ = []
    test_ = []
    # Train
    with open(train_filenames,'r') as file:
        train = file.readlines()
    for line in train:
        train_.append((line.split('.')[0]))
    # Test
    with open(test_filenames,'r') as file:
        test = file.readlines()
    for line in test:
        test_.append((line.split('.')[0]))
    # Val 
    with open(val_filenames,'r') as file:
        val = file.readlines()
    for line in val:
        val_.append((line.split('.')[0]))
    
    return train_,val_,test_

## GENERAL UTILITIES

def beam_search(model,image_features,n_pred,value_to_idx,k,device):
    # Define the softmax function
    endseq_idx = value_to_idx['endseq']

    prediction = model.sample(image_features,n_pred,endseq_idx,k,device)
    
    return prediction

def greedy_search(model,sequence,n_pred,value_to_idx):
    # Set the model in evalulation mode
    model.eval()
    # Define the softmax function
    softmax = nn.Softmax(dim=0)
    
    endseq_idx = value_to_idx['endseq']
    startseq_idx = value_to_idx['startseq']
    
    # In full_prediction we will save the complete prediction
    full_prediction = []
    
    for i in range(sequence.shape[1]):
        if(sequence[0,i].item()!=startseq_idx):
            full_prediction.append(sequence[0,i].item())
    
    for _ in range(n_pred):
        # Put batch size equals to 1 for coerence
        state = None
        # Make a prediction given the pattern
        prediction,state = model(sequence,state)
        # Remove batch
        if prediction.shape[2]==1:
            prediction = prediction.view(-1,1)
        else:
            prediction = prediction.squeeze()
        # It is applied the softmax function to the predicted tensor
        prediction = softmax(prediction[:,-1])
        # It is taken the idx with the highest probability
        arg_max = torch.argmax(prediction)
        # The window is sliced 1 character to the right
        new_seq = torch.zeros((sequence.shape[0],sequence.shape[1]+1))
        
        new_seq[0,:-1] = sequence
        new_seq[0,-1] = arg_max
        sequence = new_seq.cuda()
        # The full prediction is saved
        if(arg_max==endseq_idx):
            break
        
        full_prediction.append(arg_max.cpu().numpy().item())
    
    return full_prediction

def convert_sentence_to_idx_array(sequence, word_to_idx):
    pieces = sequence.split(' ')
    array = np.zeros((len(pieces)))
    for i in range(len(pieces)):
        array[i] = word_to_idx[pieces[i]]
    
    array = torch.from_numpy(array)
    
    if(torch.cuda.is_available()):
        array = array.cuda()
        
    return array


## METRICS ##
# Introduction on perplexity, cross entropy and bits per symbol https://thegradient.pub/understanding-evaluation-metrics-for-language-models/#fn10
#from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def evaluate_model(descriptions, predicted_desciptions):
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(),"ROUGE_L"),
            (meteor_score, "METEOR"),
            (Cider(), "CIDEr")
        ]
    for scorer, name in scorers:
        print(name)
        if(name=="METEOR"):
            meteor_scores = []
            for key in descriptions.keys():
                references = [caption.split(' ') for caption in descriptions[key]]
                prediction = predicted_desciptions[key][0].split(' ')
                meteor_scores.append(scorer(references,prediction))
            score = np.mean(meteor_scores)
            print(score)
        else:
            score, _ = scorer.compute_score(descriptions, predicted_desciptions)
            if(type(score) == list):
                score = ' '.join([str(round(sc*100,2)) for sc in score])
                print(score)
            else:
                print(score)
        
    # # calculate BLEU score
    # print('BLEU-1: %f' % bleu1)
    # print('BLEU-2: %f' % bleu2)
    # print('BLEU-3: %f' % bleu3)
    # print('BLEU-4: %f' % bleu4)
    
    # return bleu1,bleu2,bleu3,bleu4
    
    return
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',monitor='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.monitor = monitor
        
    def __call__(self, score, model):
        if(self.monitor=='min'):
            score_temp = -score
        elif(self.monitor=='max'):
            score_temp = score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
            
        elif score_temp < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score_temp
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose and self.monitor=='min':
            print(f'Score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        elif self.verbose and self.monitor=='max':
            print(f'Score increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def generator(model, image_features, idx_to_value, value_to_idx, n_pred, k, device):
    
    full_prediction = beam_search(model,image_features,n_pred,value_to_idx,k, device)
    endseq_idx = value_to_idx['endseq']

    captions = []
    for prediction in full_prediction:
        try:
            index = prediction.index(endseq_idx)
            prediction = [idx_to_value[idx] for idx in prediction[1:index]]
        except:
            prediction = [idx_to_value[idx] for idx in prediction[1:]]

        captions.append(prediction)
    return captions
