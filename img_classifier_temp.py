import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tqdm

class ConvNet(nn.Module):
    def __init__(self, device):
        super(ConvNet, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(3, 16, 3) # specify the size (outer numbers) and amount (middle number) of filters
        self.pool = nn.MaxPool2d(2, 2) # specify pool size first number is size of pool, second is step size
        self.conv2 = nn.Conv2d(16, 8, 3) # new depth is amount of filters in previous conv layer
        self.fc1 = nn.Linear(54*54*8, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2) # final fc layer needs 19 outputs because we have 19 layers # ???

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 54*54*8) # flatten
        x = F.relu(self.fc1(x))    # fully connected, relu        
        x = F.relu(self.fc2(x))    
        x = self.fc3(x)     # output    
        return x
    
    def dev(self, model, val_loader): 
        model.to(self.device)
        batch_size = val_loader.batch_size
        avg = 'macro' # used when computing certain accuracy metrics
        model.eval()

        eval_loss = 0

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for b, batch in tqdm.tqdm(enumerate(val_loader), 
                                total= len(val_loader), desc=f"Processing validation data"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                raw_logits = model.forward(images)

                preds = torch.argmax(raw_logits, dim=1) # https://discuss.pytorch.org/t/cross-entropy-loss-get-predicted-class/58215

                loss = nn.CrossEntropyLoss()(raw_logits, labels)

                eval_loss += loss.item()

                all_preds.extend(preds.tolist())
                all_trues.extend(labels.tolist())


            # metrics 
            acc_total = accuracy_score(y_true=all_trues, y_pred=all_preds)
            precision = precision_score(y_true=all_trues, y_pred=all_preds, zero_division=0, average=avg)
            recall = recall_score(y_true=all_trues, y_pred=all_preds, zero_division=0, average=avg)
            f1 = f1_score(y_true=all_trues, y_pred=all_preds, zero_division=0, average=avg)

            avg_eval_loss = eval_loss / (len(val_loader))

            metrics = {
                'accuracy': acc_total, 
                'precision': precision, 
                'recall': recall, 
                'f1': f1, 
                'avg_eval_loss': avg_eval_loss
            }
            print('****Evaluation****')
            print(f'total_accuracy: {acc_total}')

            return acc_total, precision, recall, f1