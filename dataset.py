import torch
    
class Dataset_UCM_UAV:
    def __init__(self, sentences, images_idx, images, word2idx):
        # The targets are automatically inferred from the input data
        # The idea is that we shift one step to the right, concatenate img features and predict the original word based on the shifted input!
        self.sentences = sentences # a list of sentences
        self.images_idx = images_idx
        self.images = images
        self.word2idx = word2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self,idx):
        img = self.images[self.images_idx[idx]].squeeze(0)
        sentence = self.sentences[idx] 
        # Convert sentece to indexes 
        sentence_idx = []
        for word in sentence:
            try:
                sentence_idx.append(self.word2idx[word])
            except:
                pass
        
        sentence_idx = torch.Tensor(sentence_idx)
        return img, sentence_idx
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (4096).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 4096).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    captions_ = torch.zeros(len(captions), max(lengths)).long() # max lenghts is also the length of the first caption! 
    for i, cap in enumerate(captions):
        end = lengths[i]
        captions_[i, :end] = cap[:end] 
    
    return images, captions_, lengths

        