#%%
import pandas as pd


# %%
with open("content/movie_plot/movie_categories.txt", "r") as f:
    movie_categories = [line.strip() for line in f.readlines()]

cat2id = {cat: idx for idx, cat in enumerate(movie_categories)}

train_df = pd.read_csv("content/movie_plot/train_movie_plots.csv")
train_plot = train_df["movie_plot"]
train_labels = train_df["movie_category"].map(cat2id)

test_df = pd.read_csv("content/movie_plot/test_movie_plots.csv")
test_plot = test_df["movie_plot"]
test_labels = test_df["movie_category"].map(cat2id)
# %%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class NlpDataset(Dataset):
    def __init__(self,data,labels,tokenizer):
        self.data = data.to_list()
        self.labels = labels.tolist()
        self.encodings = tokenizer(self.data, truncation=True, padding=True)

    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx],dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

    
train_dataset = NlpDataset(train_plot, train_labels, tokenizer)
test_dataset = NlpDataset(test_plot,test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)
# %%
next(iter(train_loader))
# %%
from transformers import  DistilBertForSequenceClassification
import torch.nn as nn

class BertClf(nn.Module):

    def __init__(self, distilbert):

        super(BertClf, self).__init__()

        self.distilbert = distilbert
        for name, param in distilbert.named_parameters():
            if not "classifier" in name:
                param.requires_grad = False

    def forward(self, sent_id, mask):

        out = self.distilbert(sent_id, attention_mask=mask)
        logits = out.logits
        attn = out.attentions
        hidden_states = out.hidden_states


        return logits,hidden_states,attn

distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                  num_labels=len(movie_categories),
                                                                  output_attentions=True,
                                                                  output_hidden_states=True)

model = BertClf(distilbert)
# %%
from tqdm.notebook import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")


def train_bert(model, optimizer, criterion, dataloader, epochs):
  model.train()
  for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0
    total = 0
    t = tqdm(dataloader)
    for i, batch in enumerate(t):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        preds, _, _ = model(input_ids,mask=attention_mask)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = preds.max(1)
        running_corrects += predicted.eq(labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

        t.set_description(f"epoch:{epoch} loss: {(running_loss / (i+1)):.4f} current accuracy:{round(running_corrects / total * 100, 2)}%")

def test_bert(model, dataloader):
    model.eval()
    test_corrects = 0
    total = 0
    with torch.no_grad():
      for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds, _, _ = model(input_ids,mask=attention_mask)
        _, predicted = preds.max(1)
        test_corrects += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return test_corrects / total

