import os
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import column_or_1d
from torch.utils.data import Dataset
import torch

# make path to home directory
home = os.path.expanduser("~")

# CLASSES
class ClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, target1_encoder, target2_encoder, device):
        self.df = df
        self.tokenizer = tokenizer
        self.target1_encoder = target1_encoder
        self.target2_encoder = target2_encoder
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get raw text and labels
        text = self.df.iloc[idx]['raw_words']
        target1 = self.df.iloc[idx]['target1']
        target2 = self.df.iloc[idx]['target2']

        # Tokenize text
        encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze(0).to(self.device)  # remove the batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0).to(self.device)

        # Get sequence length (this is needed for your LSTM)
        seq_len = input_ids.size(0)

        # Return all necessary items for the model
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'seq_len': seq_len,
            'target1': torch.tensor(target1, dtype=torch.long).to(self.device),  # Ensure it's a tensor of long integers
            'target2': torch.tensor(target2, dtype=torch.long).to(self.device),  # Ensure it's a tensor of long integers
        }

class CustomLabelEncoder(LabelEncoder):
    '''
    Modifies the sklearn LabelEncoder to use the inputted labels without sorting them
    '''
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self
    def fit_transform(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        
        return [self.classes_.tolist().index(item) for item in y]
    def transform(self, y):
        return [self.classes_.tolist().index(item) for item in y]
    
# FUNCTIONS
def replace_label(df, label, replacement, label_col):
    df[label_col] = df[label_col].apply(lambda x: x if x != label else replacement)



def data_process(device, tar1_labels=None, tar2_labels=None):
    '''Used for processing the scmc dataset--a CrisisMMD dataset.
    Args:
        device: torch device to use
        tar1_labels: list of labels for target1 (classification)
        tar2_labels: list of labels for target2 (tagging) '''


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the CSV into pandas DataFrame
    data_path = Path(home, r'OneDrive - Stephen F. Austin State University\Research\Research-Resource-Allocation-Code', 
                     r'data\SCMC\scmc-main\Data\CrisisMMD_Multimodal_Crisis_Dataset', 'source', 'nxg')
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_df.columns = ['raw_words', 'target1', 'target2']
    # train_df['seq_len'] = train_df['raw_words'].apply(lambda seq: len(tokenizer.encode(seq, add_special_tokens=True)))

    dev_df = pd.read_csv(os.path.join(data_path, 'dev.csv'))
    dev_df.columns = ['raw_words', 'target1', 'target2']
    # dev_df['seq_len'] = dev_df['raw_words'].apply(lambda seq: len(tokenizer.encode(seq, add_special_tokens=True)))


    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_df.columns = ['raw_words', 'target1', 'target2']
    # test_df['seq_len'] = test_df['raw_words'].apply(lambda seq: len(tokenizer.encode(seq, add_special_tokens=True)))

    
    # reducing to 6 classes
    df_list = [train_df, dev_df, test_df]
    for df in df_list:
        replace_label(df, 'vehicle_damage', 'other_relevant_information', 'target2')
        replace_label(df, 'missing_or_found_people', 'other_relevant_information', 'target2')
    
    
    target1_encoder = CustomLabelEncoder()
    target2_encoder = CustomLabelEncoder()
    

    # Fit encoders on the labels
    if tar1_labels == None and tar2_labels == None:
        target1_encoder.fit(train_df['target1'].unique())
        target2_encoder.fit(train_df['target2'].unique())
    elif tar1_labels == None:
        target1_encoder.fit(train_df['target1'].unique())
        target2_encoder.fit(tar2_labels)
    else:
        target1_encoder.fit(tar1_labels)
        target2_encoder.fit(tar2_labels)
    
    # Transform the labels into integer indices
    train_df['target1'] = target1_encoder.transform(train_df['target1'])
    train_df['target2'] = target2_encoder.transform(train_df['target2'])
    dev_df['target1'] = target1_encoder.transform(dev_df['target1'])
    dev_df['target2'] = target2_encoder.transform(dev_df['target2'])
    test_df['target1'] = target1_encoder.transform(test_df['target1'])
    test_df['target2'] = target2_encoder.transform(test_df['target2'])

    
    # Creating custom datasets
    train_dataset = ClassificationDataset(train_df, tokenizer, target1_encoder, target2_encoder, device=device)
    dev_dataset = ClassificationDataset(dev_df, tokenizer, target1_encoder, target2_encoder, device=device)
    test_dataset = ClassificationDataset(test_df, tokenizer, target1_encoder, target2_encoder, device=device)

    return train_dataset, dev_dataset, test_dataset, target1_encoder, target2_encoder

def calculate_class_weights(dataset, num_classes, device):
    class_counts = np.zeros(num_classes)
    for instance in dataset:
        # get the class of the instance, which indexes class_counts
        class_counts[instance['target2']] += 1
        
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights)
    return torch.tensor(class_weights, dtype=torch.float32).to(device)

def total_acc(pred_tar1, true_tar1, pred_tar2, true_tar2):
    total_count, correct_count = 0.0, 0.0
    for p_class, r_class, p_tag, r_tag in zip(pred_tar1, true_tar1, pred_tar2, true_tar2):
        if p_class == r_class and p_tag == r_tag:
            correct_count += 1.0
        total_count += 1.0
    return 1.0 * correct_count / total_count  
