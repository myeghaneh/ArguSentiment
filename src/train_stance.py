import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from itertools import chain


# -----------------------------
# Dataset
# -----------------------------
class StanceDataset(Dataset):
    def __init__(self, texts, stances, topics, tokenizer, max_len, extra_feats=None, sample_level=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.sample_level = sample_level

        if sample_level:
            for text_list, stance_list, feats_list, topic in zip(texts, stances, extra_feats, topics):
                for i in range(len(text_list)):
                    self.samples.append({
                        'text': text_list[i],
                        'stance': float(stance_list[i]),
                        'topic': topic,
                        'extra_feat': feats_list[i] if feats_list is not None else None
                    })
        else:
            for text, stance, topic, feat in zip(texts, stances, topics, extra_feats if extra_feats is not None else [None] * len(texts)):
                self.samples.append({
                    'text': text,
                    'stance': stance,
                    'topic': topic,
                    'extra_feat': feat
                })

        self.topic_vocab = {topic: idx for idx, topic in enumerate(set(s['topic'] for s in self.samples))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer.encode_plus(
            sample['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        extra_feat = sample['extra_feat']
        if extra_feat is not None:
            extra_feat = torch.tensor(extra_feat, dtype=torch.float)
        else:
            extra_feat = torch.tensor([], dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stance': torch.tensor(sample['stance']),
            'topic': torch.tensor(self.topic_vocab[sample['topic']], dtype=torch.long),
            'extra_feat': extra_feat
        }


# -----------------------------
# Model
# -----------------------------
class StanceClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', extra_feat_dim=0, num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.extra_feat_dim = extra_feat_dim

        hidden_size = self.bert.config.hidden_size
        input_dim = hidden_size + extra_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, extra_feats=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        if self.extra_feat_dim > 0 and extra_feats is not None and extra_feats.numel() > 0:
            x = torch.cat((pooled_output, extra_feats), dim=1)
        else:
            x = pooled_output

        return self.classifier(x)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, data_loader, device, loss_fn, is_binary=True):
    model.eval()
    eval_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_feats = batch["extra_feat"].to(device) if batch["extra_feat"].numel() > 0 else None

            if is_binary:
                stances = batch["stance"].to(device).unsqueeze(1).float()
            else:
                stances = batch["stance"].to(device).long()

            outputs = model(input_ids, attention_mask, extra_feats)
            loss = loss_fn(outputs, stances)
            eval_loss += loss.item()

            if is_binary:
                preds = (torch.sigmoid(outputs) > 0.5).int().flatten()
                labels = stances.int().flatten()
            else:
                preds = torch.argmax(outputs, dim=1).flatten()
                labels = stances.flatten()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = eval_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary' if is_binary else 'macro')
    precision = precision_score(all_labels, all_preds, zero_division=0, average='binary' if is_binary else 'macro')
    recall = recall_score(all_labels, all_preds, zero_division=0, average='binary' if is_binary else 'macro')

    return avg_loss, f1, precision, recall, all_preds


# -----------------------------
# Training
# -----------------------------
def train_model(X_train, y_train, X_test, y_test,
                batch_size=4, epochs=6, device="cpu",
                model_name="bert-base-uncased", sample_level=False,
                is_binary=None, saved_model_path="trained_stance_model.pt"):

    from itertools import chain

    # -------- Prepare input data --------
    if isinstance(X_train, (tuple, list)):
        texts_train, topics_train = X_train[0], X_train[1]
        feats_train = X_train[2] if len(X_train) > 2 else None
        texts_test, topics_test = X_test[0], X_test[1]
        feats_test = X_test[2] if len(X_test) > 2 else None
    else:
        texts_train = X_train["text"].tolist()
        topics_train = X_train["topic_id"].tolist()
        feats_train = X_train.get("nrc_feats", None)
        texts_test = X_test["text"].tolist()
        topics_test = X_test["topic_id"].tolist()
        feats_test = X_test.get("nrc_feats", None)

    y_train = y_train.tolist() if hasattr(y_train, "tolist") else y_train
    y_test = y_test.tolist() if hasattr(y_test, "tolist") else y_test

    # -------- Handle label format --------
    y_train_flat = list(chain.from_iterable(y_train)) if sample_level else y_train
    unique_labels = sorted(set(y_train_flat))
    if is_binary is None:
        is_binary = len(unique_labels) == 2  # safer to set manually
    num_classes = 1 if is_binary else len(unique_labels)

    extra_feat_dim = 0
    if feats_train is not None:
        sample = feats_train[0][0] if sample_level else feats_train[0]
        extra_feat_dim = len(sample) if isinstance(sample, (list, tuple)) else 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = StanceDataset(texts_train, y_train, topics_train, tokenizer, 128,
                                  extra_feats=feats_train, sample_level=sample_level)
    test_dataset = StanceDataset(texts_test, y_test, topics_test, tokenizer, 128,
                                 extra_feats=feats_test, sample_level=sample_level)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = StanceClassifier(model_name, extra_feat_dim, num_classes).to(device)
    print("Model architecture:")
    print(model)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_loss': [], 'train_f1': [], 'test_f1': [], 'test_precision': [], 'test_recall': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_train_preds, all_train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_feats = batch["extra_feat"].to(device) if batch["extra_feat"].numel() > 0 else None

            if is_binary:
                stances = batch["stance"].to(device).unsqueeze(1).float()
            else:
                stances = batch["stance"].to(device).long()

            outputs = model(input_ids, attention_mask, extra_feats)
            loss = loss_fn(outputs, stances)
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                if is_binary:
                    preds = (torch.sigmoid(outputs) > 0.5).int().flatten()
                    labels = stances.int().flatten()
                else:
                    preds = torch.argmax(outputs, dim=1).flatten()
                    labels = stances.flatten()

                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0, average='binary' if is_binary else 'macro')

        test_loss, test_f1, test_precision, test_recall, y_test_pred = evaluate_model(
            model, test_loader, device, loss_fn, is_binary=is_binary
        )

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss)
        history['train_f1'].append(train_f1)
        history['test_f1'].append(test_f1)
        history['test_precision'].append(test_precision)
        history['test_recall'].append(test_recall)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Test Loss: {test_loss:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print("-" * 50)

    # Save everything
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': model_name,
            'extra_feat_dim': extra_feat_dim,
            'num_classes': num_classes
        },
        'tokenizer_name': model_name,
        'topic_vocab': train_dataset.topic_vocab,
        'history': history,
        'y_test': y_test,
        'y_pred': y_test_pred
    }

    torch.save(save_dict, saved_model_path)
    print(f"✅ Full model checkpoint saved to: {saved_model_path}")

    return model, history, y_test_pred
