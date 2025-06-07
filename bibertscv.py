import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import torch.nn.utils.rnn as rnn_utils
import copy


def focal_loss_multiclass(inputs, targets, alpha=1, gamma=2):
    """
    Multi-class focal loss implementation
    - inputs: raw logits from the model
    - targets: true class labels (as integer indices, not one-hot encoded)
    """
    # Convert logits to log probabilities
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)  # Calculate probabilities from log probabilities

    # Gather the probabilities corresponding to the correct classes
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    pt = torch.sum(prob * targets_one_hot, dim=-1)

    #  focal adjustment
    focal_loss = -alpha * (1 - pt) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)
    
    return focal_loss.mean()

def bibert_pipeline(tweet_dicts, tokenizer, bibert, target1_id2label, 
                    target2_id2label, device='cpu', custom_batch_size=16):
    """
    Processes a list of tweet dictionaries through a BiBERT model pipeline, assigning predicted labels and tags.
    Args:
        tweet_dicts (list of dict): List of dictionaries, each containing a 'tweet' key with the tweet text.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to preprocess tweet texts.
        bibert (object): BiBERT model object with a `predict` method.
        target1_id2label (dict): Mapping from target 1 class indices to label names.
        target2_id2label (dict): Mapping from target 2 class indices to tag names.
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        custom_batch_size (int, optional): Number of tweets to process per batch. Defaults to 16.
    Returns:
        list of dict: Deep copy of input tweet_dicts, with added 'label' and 'tag' keys for each tweet, containing the predicted class and tag.
    """

    labeled_dicts = copy.deepcopy(tweet_dicts)
    
    batch_size = custom_batch_size
    num_full_batches = len(tweet_dicts) // batch_size # if you have 33 tweets, this would be 2
    last_batch_len = len(tweet_dicts) % batch_size # if you have 33 tweets, this would be 1

    for i in range(num_full_batches):
        tweets = [tweet_dict['tweet'] for tweet_dict in tweet_dicts[i*batch_size:i*batch_size+batch_size]]
        tokenized_inputs = tokenizer(tweets, padding=True, truncation=True, return_tensors='pt')
        tokenized_inputs.to(device)
        seq_len = tokenized_inputs['attention_mask'].sum(dim=1).to('cpu')
    
        informative_pred, tag_pred = bibert.predict(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], seq_len, flags=2)
    
        informative_pred = informative_pred
        tag_pred = tag_pred
    
        # assert(len(tweets) == len(informative_pred))

        for t, tweet in enumerate(tweets): # 16 tweets, each tweet from batch i 
            loc = t + i*batch_size # this gets the t tweet from the current ith batch # if t is 1, we're on batch 1 so i is 1, then this would be 1 + 16 = 17
            labeled_dicts[loc]['label'] = target1_id2label[int(informative_pred[t])]
            labeled_dicts[loc]['tag'] = target2_id2label[int(tag_pred[t])]

    if last_batch_len > 0:
        tweets = [tweet_dict['tweet'] for tweet_dict in tweet_dicts[(num_full_batches-1)*batch_size : (num_full_batches-1)*batch_size + last_batch_len]]
        tokenized_inputs = tokenizer(tweets, padding=True, truncation=True, return_tensors='pt')
        tokenized_inputs.to(device)
        seq_len = tokenized_inputs['attention_mask'].sum(dim=1).to('cpu')
    
        informative_pred, tag_pred = bibert.predict(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], seq_len, flags=2)
    
        informative_pred = informative_pred
        tag_pred = tag_pred
        for t, tweet in enumerate(tweets): # 16 tweets, each tweet from batch i 
            loc = t + (num_full_batches - 1)*batch_size 
            labeled_dicts[loc]['label']  = target1_id2label[int(informative_pred[t])]
            labeled_dicts[loc]['tag'] = target2_id2label[int(tag_pred[t])]
    
            
    return labeled_dicts
    

# The below model is adapted from the paper and code from this project: https://github.com/SCMCmodel/scmc
# TODO: make this predict informative/non-informative before humanitarian classes
class BibertSCV(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2, num_tags=6, hidden_size=256, 
                 temperature=1, num_layers=1, dropout=0.3, label_class_weights=None, tag_class_weights=None, fl_gamma=2, fl_alpha=1):
        super(BibertSCV, self).__init__()
        self.temperature = temperature
        self.num_labels = num_labels
        self.num_tags = num_tags
        self.bert = BertModel.from_pretrained(model_name)
        self.m = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        self.critrion_class = nn.MultiLabelSoftMarginLoss()
        self.critrion_tag = nn.MultiLabelSoftMarginLoss()
        self.ce_class = nn.CrossEntropyLoss(weight=label_class_weights)
        self.ce_tag = focal_loss_multiclass
        self.ce_tag_alpha=fl_alpha
        self.ce_tag_gamma=fl_gamma

        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers, bidirectional=True)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.fc_class = nn.Linear(in_features=hidden_size, out_features=num_tags)
        self.fc_tag = nn.Linear(in_features=hidden_size, out_features=num_tags)
        self.maxpooling = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, input_ids, attention_mask, flags, seq_len=None, target1=None, target2=None):
        '''
        Parameters:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            seq_len: [batch_size] (sequence lengths for padding handling)
            target1: [batch_size] (labels for classification)
            target2: [batch_size] (labels for tagging)
            flags: 0 means training, 2 means prediction
        Return:
            logits: torch.Tensor [batch_size, num_labels]
        '''
        if flags==0 and (seq_len is None or target1 is None or target2 is None):
            raise ValueError("seq_len, target1, and target2 must be provided for training mode (flags=0)")
        
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Get the BERT hidden states
        input_x = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]  # [batch_size, seq_len, hidden_size]

        # # CLS embedding (first token's embedding)
        # cls_embed = input_x[:, 0, :]  # Shape: [batch_size, hidden_size]

        # Apply dropout
        hidden = self.dropout(input_x)

        packed_input = rnn_utils.pack_padded_sequence(hidden, seq_len, batch_first=True, enforce_sorted=False)

        # Pass through LSTM layer
        packed_outputs, (_, _) = self.lstm(packed_input)
    
        outputs, _ = rnn_utils.pad_packed_sequence(packed_outputs, batch_first=True)

        # Apply dropout layer
        outputs = self.dropout_layer(outputs)
        
        # Summarizes the LSTM output by taking the maximum activation per feature across all tokens.
        feat = self.maxpooling(outputs.transpose(1, 2)).squeeze(2)  # [batch_size, hidden_size]

        # Classification and Tagging output logits
        out_class = self.fc_class(feat)  # logits for class
        out_tag = self.fc_tag(feat)      # logits for tagging

        if flags == 0:  # Training mode
            loss_class_c, loss_tag_c = self.scl(feat, seq_len, target1, target2, flags)
            loss_class = self.ce_class(out_class, target1)
            loss_tag = self.ce_tag(out_tag, target2, alpha=self.ce_tag_alpha, gamma=self.ce_tag_gamma)

            return loss_class_c, loss_tag_c, loss_class, loss_tag

        if flags == 2:  # Prediction mode
            class_pre = torch.max(out_class, -1)[1]
            tag_pre = torch.max(out_tag, -1)[1]

            return class_pre, tag_pre, out_class, out_tag 

    def scl(self, feature, seq_len, target1, target2, flags):
        feature_x = self.m(feature) # batch norm
        temp_feature = torch.matmul(feature_x, feature.permute(1, 0)) 
        logit = torch.divide(temp_feature, self.temperature)
        loss_class, loss_tag = self.scl_loss(logit, seq_len, target1, target2, flags)
        return loss_class, loss_tag
    
    def scl_loss(self, logit, seq_len, target1, target2, flags):
        class_pred = logit
        class_true = target1.type_as(class_pred)

        tag_pred = logit
        tag_true = target2.type_as(tag_pred)

        # Similarity calculations for class and tag predictions
        # .eq computes element-wise equality
        class_true_x = torch.unsqueeze(class_true, -1)
        class_true_y = (torch.eq(class_true_x, class_true_x.permute(1, 0))).type_as(class_pred)
        class_true_z = class_true_y / torch.sum(class_true_y, 1, keepdim=True)

        tag_true_x = torch.unsqueeze(tag_true, -1)
        tag_true_y = torch.eq(tag_true_x, tag_true_x.permute(1, 0)).type_as(tag_pred)
        tag_true_z = tag_true_y / torch.sum(tag_true_y, 1, keepdim=True)

        # Cross entropy loss for class and tag predictions
        class_cross_entropy = self.critrion_class(class_pred, class_true_z)
        tag_cross_entropy = self.critrion_class(tag_pred, tag_true_z)

        return class_cross_entropy, tag_cross_entropy

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask, flags=2)
        if self.num_labels > 1:
            return torch.argmax(logits, dim=-1)
        else:
            return logits