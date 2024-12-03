import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import os
import numpy as np
import pandas as pd
import wave
import librosa
import re
import os
import itertools



a=np.load("/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train/train_audio_feats.npz", allow_pickle=True)['audio_features']
b= np.load("/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/test_audio_feats.npz", allow_pickle=True)['audio_features']

t1=[np.squeeze(np.array(audio), axis=1) for audio in a]
t2 = [
    np.squeeze(np.array(audio), axis=1)
    for audio in b
    if len(np.array(audio).shape) == 3
]

train_audio_features=t1
test_audio_features=t2

c=np.load("/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train/train_text_feats.npz", allow_pickle=True)['text_features']
d= np.load("/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/test__text_feats.npz", allow_pickle=True)['text_features']

t11=[np.squeeze(np.array(text), axis=1) for text in c]
t22 = [
    np.squeeze(np.array(text), axis=1)
    for text in d
    if len(np.array(text).shape) == 3
]

train_text_features=t11
test_text_features=t22
train_labels = np.load("/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/train/train_c_labels.npz", allow_pickle=True)['labels']
test_labels = np.load("/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Features/test/test_c_labels.npz", allow_pickle=True)['labels']

# Remove labels at indices 24 and 25
indices_to_remove = [24, 25]
test_labels = np.delete(test_labels, indices_to_remove, axis=0)


train_audio_features_tensors = [torch.tensor(f) for f in train_audio_features]
train_audio_features = pad_sequence(train_audio_features_tensors, batch_first=True)

test_audio_features_tensors = [torch.tensor(f) for f in test_audio_features]
test_audio_features = pad_sequence(test_audio_features_tensors, batch_first=True)


train_text_features_tensors = [torch.tensor(f) for f in train_text_features]
train_text_features = pad_sequence(train_text_features_tensors, batch_first=True)


test_text_features_tensors = [torch.tensor(f) for f in test_text_features]
test_text_features = pad_sequence(test_text_features_tensors, batch_first=True)

train_labels = np.array(train_labels, dtype=np.int64)
test_labels = np.array(test_labels, dtype=np.int64)


########### data aug #############

from imblearn.over_sampling import SMOTE
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels_t = np.squeeze(train_labels)  # Converts to 1D if it's in shape (n_samples, 1)

# Flatten the audio features to 2D (n_samples, sequence_length * num_features)
flattened_train_text_features = train_text_features.view(train_text_features.size(0), -1)
#print(flattened_train_text_features.shape)  # Should be [n_samples, 63 * 256]

# Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
balanced_text_features, balanced_labels = smote.fit_resample(flattened_train_text_features.numpy(), labels_t)

# Convert the balanced features and labels to PyTorch tensors
balanced_text_features = torch.tensor(balanced_text_features).to(device)
balanced_labels = torch.tensor(balanced_labels).to(device)
# Check the current total size and calculate time steps
num_samples = balanced_text_features.size(0)
num_features = 1024  # Assuming this is fixed
time_steps = balanced_text_features.size(1) // num_features

# Reshape to 3D
balanced_text_features = balanced_text_features.view(num_samples, time_steps, num_features)
#print(balanced_text_features.shape)  # Verify the shape


# Flatten the audio features to 2D (n_samples, sequence_length * num_features)
flattened_train_audio_features = train_audio_features.view(train_audio_features.size(0), -1)
#print(flattened_train_audio_features.shape)  # Should be [n_samples, 63 * 256]

# Apply SMOTE to balance the data
smote = SMOTE(random_state=42)
balanced_audio_features, balanced_labels = smote.fit_resample(flattened_train_audio_features.numpy(), labels_t)

# Convert the balanced features and labels to PyTorch tensors
balanced_audio_features = torch.tensor(balanced_audio_features).to(device)
balanced_labels = torch.tensor(balanced_labels).to(device)
# Check the current total size and calculate time steps
num_samples = balanced_audio_features.size(0)
num_features = 256  # Assuming this is fixed
time_steps = balanced_audio_features.size(1) // num_features

# Reshape to 3D
balanced_audio_features = balanced_audio_features.view(num_samples, time_steps, num_features)
#print(balanced_audio_features.shape)  # Verify the shape


class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.num_classes = config['num_classes']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        self.hidden_dims = config['hidden_dims']
        self.rnn_layers = config['rnn_layers']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']

        self.build_model()
        self.init_weight()
        
    def init_weight(net):
        for name, param in net.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def build_model(self):
        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.dropout,
                                bidirectional=self.bidirectional)
        
        # self.init_weight()
        
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims, self.num_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # h = lstm_out
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        
        # x : [len_seq, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(x)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)
    
    
config = {
        'num_classes': 2,
        'dropout': 0.4,
        'rnn_layers': 2,
        'audio_embed_size': 256,
        'text_embed_size': 1024,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 5e-4,
        'audio_hidden_dims': 256,
        'text_hidden_dims': 128,
        'cuda': False,
        'lambda': 1e-5,
    }

class FusionDataset(Dataset):
    def __init__(self, audio_features, text_features, labels):
        self.audio_features = audio_features
        self.text_features = text_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.audio_features[idx]
        text = self.text_features[idx]
        label = self.labels[idx]

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.long).squeeze()
    

train_dataset = FusionDataset(balanced_audio_features, balanced_text_features, balanced_labels)
test_dataset = FusionDataset(test_audio_features, test_text_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


class FusionNet(nn.Module):
    def __init__(self, config):
        super(FusionNet, self).__init__()
        self.text_lstm = nn.LSTM(
            input_size=config['text_embed_size'],
            hidden_size=config['text_hidden_dims'],
            num_layers=config['rnn_layers'],
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )
        self.text_fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['text_hidden_dims'], config['text_hidden_dims']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(config['text_hidden_dims'], config['text_hidden_dims']),
            nn.ReLU(inplace=True)
        )
        
        
        self.audio_gru = nn.GRU(
            input_size=config['audio_embed_size'],
            hidden_size=config['audio_hidden_dims'],
            num_layers=config['rnn_layers'],
            dropout=config['dropout'],
            bidirectional=False,
            batch_first=True
        )
        self.audio_fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['audio_hidden_dims'], config['audio_hidden_dims']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        self.fc_final = nn.Sequential(
            nn.Linear(config['text_hidden_dims'] + config['audio_hidden_dims'], config['num_classes']),
            # nn.Softmax(dim=1)
        )
        
        self.ln = nn.LayerNorm(config['audio_hidden_dims'])
        
        self.modal_attn = nn.Linear(config['text_hidden_dims'] + config['audio_hidden_dims'], config['text_hidden_dims'] + config['audio_hidden_dims'], bias=False)

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        # print(f"lstm_out: {lstm_out.shape}")
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # print(f"lstm out shape: {lstm_tmp_out.shape}")
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # print(f"h: {h.shape}")
        # [batch_size, num_layers * num_directions, n_hidden]
        # print(f"lstm_hidden: {lstm_hidden.shape}")
        lstm_hidden = torch.sum(h, dim=1)
        # print(f"lstm_hidden: {lstm_hidden.shape}")
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        # print(lstm_hidden.shape)
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        # print(f"m: {m.shape}")
        # print(f"atten_w: {atten_w.shape}")
        # m = m.squeeze(1)
        m = m.transpose(1, 2)
        # print(f"m: {m.shape}")
        atten_context = torch.bmm(atten_w, m)
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result
    
    def forward(self, audio, text):
        text_out, (final_hidden_state, _) = self.text_lstm(text)
        # text_out = text_out[:, -1, :]
        # print(f"text out dim: {text_out.shape}")
        # text_out = text_out.unsqueeze(1)
        # text_out = text_out.permute(1, 0, 2)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(text_out, final_hidden_state)
        # print(f"atten_out: {atten_out.shape}")
        text_features = self.text_fc(atten_out)
        
        

        audio_out, _ = self.audio_gru(audio)
        #audio_out = audio_out[:, -1, :]
        audio_out = audio_out.mean(dim=1)  # Sum across time steps (dim=1)

        audio_features = self.audio_fc(audio_out)

        combined_features = torch.cat((text_features, audio_features), dim=1)
        # print(f"text_features shape: {text_features.shape}")  # Expected: [batch_size, text_hidden_dims]
        # print(f"audio_features shape: {audio_features.shape}")  # Expected: [batch_size, audio_hidden_dims]
        # print(f"combined_features shape: {combined_features.shape}")  # Expected: [batch_size, text_hidden_dims + audio_hidden_dims]
        modal_weights = torch.softmax(self.modal_attn(combined_features), dim=1)
        # modal_weights = self.modal_attn(combined_features)
        combined_features = (modal_weights * combined_features)
        output = self.fc_final(combined_features)
        return output

device = torch.device("cuda" if config['cuda'] and torch.cuda.is_available() else "cpu")
model = FusionNet(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['lambda'])

def train(train_loader,val_loader,epoch,best_accuracy):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for audio, text, labels in train_loader:
        audio, text, labels = audio.to(device), text.to(device), labels.to(device)

        # print(f"Audio batch shape: {audio.shape}")
        #print(f"Text batch shape: {text.shape}")
        # print(audio.shape)
        # print(text.shape)
        outputs = model(audio, text)  # This will trigger the forward method
        #print("Forward pass successful!")
        
        optimizer.zero_grad()
        #outputs = model(audio, text)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for audio, text, labels in train_loader:
            audio, text, labels = audio.to(device), text.to(device), labels.to(device)
            outputs = model(audio, text)  # This will trigger the forward method
        
            #outputs = model(audio, text)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pred = outputs.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / len(val_loader.dataset)

    print(f"Train Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), '/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Classification/Models/fusion_best_model_4.pth')  # Save the best model
    return best_accuracy

''''
best_accuracy=0.0
for epoch in range(config['epochs']):
        best_accuracy = train(train_loader,test_loader,epoch,best_accuracy)
print(f"Best model saved with accuracy: {best_accuracy:.4f}")
     ''' 

def evaluate(test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for audio, text, labels in test_loader:
            audio, text, labels = audio.to(device), text.to(device), labels.to(device)
            outputs = model(audio, text)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        average_loss = total_loss / len(test_loader)
        
        results_df = pd.DataFrame({
        'Predicted': predictions,
        'Target': true_labels
        })
    
        print("\nPredicted vs Target:")
        print(results_df)
        
        f1 = f1_score(true_labels, predictions, average="weighted")
        cm = confusion_matrix(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        
    print(f"Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    print("Confusion Matrix:")
    print(cm)
    return results_df


model.load_state_dict(torch.load('/mnt/sd1/jhansi/interns/chaithra/MS/sal_project/Classification/Models/fusion_best_model_4.pth', weights_only=True))
model.eval()  # Set to evaluation mode

# Evaluate on the test set
accuracy = evaluate(test_loader)       