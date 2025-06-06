import os
import pandas as pd
from pathlib import Path
from transformers import BertTokenizer

# make path to home directory
home = os.path.expanduser("~")


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