import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AdamW, get_linear_schedule_with_warmup

bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# for model.predict, it is required to define those values.
MAX_LEN = 40 
THRESHOLD = 0.6

class BERTClass(torch.nn.Module):
    def __init__(self, n_train_steps, n_classes, dropout):
        super(BERTClass, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_classes)
        self.n_train_steps = n_train_steps
        self.step_scheduler_after = "batch"
        
    def forward(self, ids, mask):

        hidden_state =  self.bert(input_ids=ids, attention_mask=mask)[0]

        pooled_output = hidden_state[:, 0]  

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits

    def fit(self, train_dataloader):

        def loss_fn(outputs, targets):
            criterion = nn.BCEWithLogitsLoss()
            criterion = criterion.to(DEVICE)
            loss = criterion(outputs.view(-1, N_CLASSES), 
                          targets.float().view(-1, N_CLASSES))
            if targets is None:
                return None
            return loss

        optimizer = torch.optim.AdamW(params =  self.parameters(), lr=LEARNING_RATE)

        def ret_scheduler(optimizer, num_train_steps):
            sch = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)
            return sch

        scheduler = ret_scheduler(optimizer, self.n_train_steps)

        def epoch_time(start_time, end_time):
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            return elapsed_mins, elapsed_secs

        for epoch in range(N_EPOCHS):
            train_loss = 0.0
            self.train()  # Set the model to training mode

            for bi, d in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                ids = d["ids"]
                mask = d["mask"]
                token_ids = d["token_ids"]
                targets = d["labels"]

                ids = ids.to(DEVICE, dtype=torch.long)
                mask = mask.to(DEVICE, dtype=torch.long)
                token_ids = token_ids.to(DEVICE,dtype=torch.long)
                targets = targets.to(DEVICE, dtype=torch.float)

                optimizer.zero_grad()
                outputs = self(ids=ids, mask=mask)
                
                loss = loss_fn(outputs, targets)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                scheduler.step()

                self.zero_grad()

            print(train_loss/len(train_dataloader))

        return train_loss/len(train_dataloader)

    def predict(self, sentence):
        max_len = MAX_LEN

        inputs = tokenizer.__call__(sentence,
                            None,
                            add_special_tokens=True,
                            max_length=max_len,
                            padding="max_length",
                            truncation=True,
                            )
        
        ids = inputs['input_ids']
        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        mask = inputs['attention_mask']
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        self.eval()
        logits = self(ids=ids, mask=mask)
        result = torch.sigmoid(logits)

        threshold = THRESHOLD
        valid_result = torch.ceil(result-threshold)

        return result, valid_result