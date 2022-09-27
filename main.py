import torch
import torch.nn as nn
import torch.optim as optim
from model import TextGenerator
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *

from tqdm import tqdm

from dataset import Dataset_UCM_UAV
from dataset import collate_fn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    args = parameter_parser()
    
    dataset = args.dataset
    feature_extractor = args.feature_extractor
    model_type = args.model_type
    
    if(torch.cuda.is_available()):
        device = torch.device('cuda:0')
    else:
        device = torch.device("cpu")
    
    # Load UCM dataset
    senteces_path = dataset+"_dataset/filenames/descriptions_"+dataset+".txt"
    train_filenames_path = dataset+"_dataset/filenames/filenames_train.txt"
    val_filenames_path = dataset+"_dataset/filenames/filenames_val.txt"
    test_filenames_path = dataset+"_dataset/filenames/filenames_test.txt"
    
    # Path to save and load weights 
    path_weights = "weights/"+dataset+"/textGenerator_"+model_type+"_"+feature_extractor+".pt"
    img_path = dataset+"_dataset/imgs/images_"+dataset+".pkl"
    path_prediction_txt = "prediction_"+dataset+"_"+model_type+"_"+feature_extractor+".txt"
    
    sentences, max_len = convert_to_words_ucm(senteces_path)
    # Sentences is a dictionary with key as the image name and value as the sentence
    
    # Split into train_test_val 
    train_,val_,test_ = create_lists_ucm_uav(train_filenames_path,val_filenames_path,test_filenames_path)
    
    train_sentences = [sentences[i] for i in train_]
    test_sentences = [sentences[i] for i in test_]
    val_sentences = [sentences[i] for i in val_]
    
    # Create the dictionary with train and val sentences
    value_to_idx,idx_to_value,ignored_words = word_frequency_ucm_uav(list(chain(*train_sentences, *val_sentences)), 5, test_sentences)
        
    train_sentences = []
    train_images_list = []
    for i in train_:
        for sentence in sentences[i]:
            train_sentences.append(sentence)
            train_images_list.append(i)

    # Load image feaures matrix
    with open(img_path,'rb') as file:
        images = pickle.load(file)
        first_key = list(images.keys())[0]
        img_feature_size = images[first_key].shape[0]
        
    
    trainset = Dataset_UCM_UAV(train_sentences,train_images_list,images,value_to_idx)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    
    model = TextGenerator(len(value_to_idx.keys()),args.hidden_dim,backbone=feature_extractor,type=model_type.lower(),trainable=False)
    
    early_stopping = EarlyStopping(patience=5, verbose=True,monitor='max',path=path_weights)
    
    # Define train loop 
    def train(model, trainloader, learning_rate, epochs, device):
        # Model initialization
        model = model.to(device)
            
        # Optimizer initialization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Set model in training mode
        model.train()
        loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
        # Training phase
        for epoch in range(epochs):
            epoch_loss = 0 
            for i, data in enumerate(tqdm(trainloader)):
                imgs,captions,lengths = data
                # Send to cuda
                imgs = imgs.to(device)
                captions = captions.to(device)
                # Feed the model
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                y_pred = model(captions,imgs,lengths)
                # Loss calculation
                loss = loss_fn(y_pred,targets)
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()
            
            predictions = []
            references_beam = []
            model.eval()
            
            for image in test_:
                reference = sentences[image]
                # Clean reference from startseq and endseq
                new_ref = []
                for sentence in reference:
                    new_ref.append(sentence[1:-1])
                
                # Get the features of the image
                input_img_features = images[image].to(device)
                
                # Get the features of the image
                prediction = generator(model,input_img_features,idx_to_value,value_to_idx,30,k=1,device=device)

                for pred in prediction:
                    references_beam.append(new_ref)
                    predictions.append(pred)
                
            
            bleu4 = evaluate_model(references_beam,predictions)
            
            early_stopping(bleu4,model)
            
            if(early_stopping.early_stop):
                print('Early stopping')
                break
            
            model.train()

            print("Epoch: %d,  loss: %.5f " % (epoch, epoch_loss/i))

    
    if(args.load_model):
        print('Loading model..')
        model.load_state_dict(torch.load(path_weights))
        if(torch.cuda.is_available()):
            model.to(device)
    else:
        train(model, trainloader, args.learning_rate, args.num_epochs, device=device)
        model.load_state_dict(torch.load(path_weights)) # To account for early stopping
    
    # PREDICT ON TRAINING SET 
    predictions_dict = dict()
    references_beam = dict()
    
    # EVALUATION OF THE MODEL 
    model.eval()
    
    print('Evaluating model on training set')
    for image in tqdm(train_):
        reference = sentences[image] # sentences[image]
        # Clean reference from startseq and endseq
        new_ref = []
        for sentence in reference:
            new_ref.append(' '.join(sentence[1:-1]))
        
        # Get the features of the image
        input_img_features = images[image].to(device) 
        
        # Get the features of the image
        predictions = generator(model,input_img_features,idx_to_value,value_to_idx,30,k=1,device=device)
        references_beam[image] = new_ref
        for prediction in predictions:
            try:
                predictions_dict[image].append(' '.join(prediction))
            except:
                predictions_dict[image] = [' '.join(prediction)]
    
    print('Results on the training set!')
    evaluate_model(references_beam,predictions_dict)
    
    predictions_dict = dict()
    references_beam = dict()

    print('Evaluating model on test set')
    for image in tqdm(test_):
        reference = sentences[image] # sentences[image]
        # Clean reference from startseq and endseq
        new_ref = []
        for sentence in reference:
            new_ref.append(' '.join(sentence[1:-1]))
            
        # Get the features of the image
        input_img_features = images[image].to(device) 
        
        # Get the features of the image
        predictions = generator(model,input_img_features,idx_to_value,value_to_idx,30,k=1,device=device)
        references_beam[image] = new_ref
        
        for prediction in predictions:
            try:
                predictions_dict[image].append(' '.join(prediction))
            except:
                predictions_dict[image] = [' '.join(prediction)]
    
    print('Results on the test set!')
    evaluate_model(references_beam,predictions_dict)
    
    # with open(path_prediction_txt, 'w') as file:
    #     for i in range(len(references_groups)):
    #         file.write('References image '+str(test_[i])+'\n')
    #         for ref in references_groups[i]:
    #             file.write(' '.join([i for i in ref])+'\n')
    #         file.write('Prediction\n')
    #         for pred in predictions_groups[i]:
    #             file.write(' '.join([i for i in pred])+'\n')
    #         file.write('\n')