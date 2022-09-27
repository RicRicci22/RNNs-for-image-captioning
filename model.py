from random import sample
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet152, ResNet152_Weights, vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
    
class TextGenerator(nn.ModuleList):
    def __init__(self, vocab_size, hidden_dim, type='gru', backbone = 'resnet152', pretrained_back = True, trainable=True):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        
        if(backbone=='resnet152'):
            self.weights = ResNet152_Weights.DEFAULT
            self.preprocess = self.weights.transforms()
            
            if(pretrained_back):
                self.backbone = resnet152(weights=self.weights)
            else:
                self.backbone = resnet152()
        elif(backbone=='vgg16'):
            self.weights = VGG16_Weights.DEFAULT
            self.preprocess = self.weights.transforms()
            
            if(pretrained_back):
                self.backbone = vgg16(weights=self.weights)
            else:
                self.backbone = vgg16()
        else:
            raise RuntimeError('Backbone not found!')
        
        if not trainable: 
            for _ , parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
        
        if (backbone=='resnet152'):   
            img_feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif(backbone=='vgg16'):
            img_feat_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Sequential(self.backbone.classifier[:-1])
                
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim,padding_idx=0) 
        self.dropout = nn.Dropout(0.5)
        # LSTM
        if(type=='gru'):
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=1) 
        elif(type=='lstm'):
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1) 
        else:
            print('Select RNN type')
            
        # Feature processing
        self.process_features = nn.Sequential(
            nn.Linear(img_feat_dim, self.hidden_dim),
        )
        # Linear layer
        self.linear1 = nn.Linear(hidden_dim, vocab_size)
		
    def forward(self, x, img, lengths):
        # Process the image
        img = self.preprocess(img)
        img_feat = self.backbone(img)
        img_feat = self.process_features(img_feat)
        
        # From idx to embedding
        x = x.long()
        x = self.dropout(self.embedding(x))

        embeddings = torch.cat((img_feat.unsqueeze(1), x), 1)
        
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        out,_ = self.rnn(packed)
        
        out = self.linear1(out[0])
        
        return out
    
    def sample(self, img, max_seq_len, endseq_index, k, device):
        """Generate captions for given image features using beam search."""
        # The algorithm must implement beam search to generate captions 
        # Beam search works like that
        # At the first iteration all the k prediction starts with the most probable word (usually start of seq!)
        # Other iterations build up on the same start of sequence, but the next tokens are sampled in a mutually exclusive way, so that different captions are created. 
        # For each caption, the k most probable words are selected. Then based on the sum of the probabilities up until that point, k sentences are kept, and so on until all reach end of sequence. 

        with torch.no_grad():
            sampled_ids = torch.zeros((k,max_seq_len+1),dtype=torch.long).to(device) # To store the sampled ids
            img = self.preprocess(img)
            img_feat = self.backbone(img)
            inputs = self.process_features(img_feat)
            
            states = None
            first = True
            sentences = []
            sum_probs_list = []
            for i in range(max_seq_len+1):
                hiddens, states = self.rnn(inputs, states)
                out = self.linear1(hiddens)
                out = out.sort(descending=True)
                probs = out[0].squeeze(1)
                predicted = out[1].squeeze(1)
                if(i==0):
                    # Append the first prediction
                    sampled_ids[:,i] = torch.tile(predicted[:,0],(k,))
                    inputs = self.embedding(sampled_ids[:,i]).unsqueeze(1)
                    states = torch.tile(states,(1,k,1))
                else:
                    probs = torch.softmax(probs,dim=1)
                    if(first):
                        sampled_ids[:,i] = predicted[0,:k]
                        sum_probs = probs[0,:k]
                        first = False
                    else:
                        # CHECK AND REMOVE FINISHED SENTENCES
                        mask = torch.all(sampled_ids!=endseq_index,dim=1)
                        idx_mask = (mask==True).nonzero()
                        
                        for j in range(mask.shape[0]):
                            if(not mask[j]):
                                sentences.append(sampled_ids[j,:].tolist())
                                sum_probs_list.append((sum_probs[j]).item())
                                
                        sampled_ids = sampled_ids[idx_mask,:].squeeze(1)
                        # Check to break the run
                        if(sampled_ids.shape[0]==0):
                            indexsorted = sorted(range(len(sum_probs_list)), key=lambda k: sum_probs_list[k],reverse=True)
                            sentences = [sentences[i] for i in indexsorted]
                            
                            break
                        
                        k = sampled_ids.shape[0]
                        # Get predictions and probabilities
                        predicted = predicted[idx_mask,:k].flatten()
                        probs = probs[idx_mask,:k].flatten()
                        sum_probs = sum_probs[idx_mask].squeeze(1)
                        # Get candidates 
                        candidates = torch.repeat_interleave(sampled_ids,k,dim=0)
                        candidates[:,i] = predicted
                        states = torch.repeat_interleave(states,k,dim=1)

                        sum_probs = torch.repeat_interleave(sum_probs,k,dim=0)
                        sum_probs = torch.mul(sum_probs,probs).sort(descending=True)
                        indices = sum_probs[1]
                        
                        sampled_ids=candidates[indices[:k],:]
                        
                        states = states[:,indices[:k],:]
                        sum_probs = sum_probs[0][:k]
                    
                    inputs = self.embedding(sampled_ids[:,i].unsqueeze(1))
                    
        return sentences 

