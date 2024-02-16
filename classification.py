import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict




with open('cancelled.json', 'rb') as file:
    cancelled_json = json.load(file)

with open('noncancelled.json', 'rb') as file:
    noncancelled_json = json.load(file)

cancelled_list = []

can_row = cancelled_json[1]
for row in cancelled_json[0:5]:
    if len(row['Comments']) > 0:
        print(row['Comments'][-1]['body'])




# for row in cancelled_json:
#     cancelled_list.append(row['Description'])

# noncancelled_list = []
# for row in noncancelled_json:
#     noncancelled_list.append(row['Description'])

# print(noncancelled_list[0])

# can_df = pd.DataFrame({"text": cancelled_list})
# can_df['label'] = [0] * can_df.shape[0]


# noncan_df = pd.DataFrame({"text": noncancelled_list})
# noncan_df['label'] = [1] * noncan_df.shape[0]

# final_df = pd.concat([can_df, noncan_df])


# final_df = final_df.sample(frac = 1)

# train_df, test_df = train_test_split(final_df, test_size=0.2)

# print(train_df.head())
# print(test_df.head())
# train_dataset = Dataset.from_pandas(train_df)
# test_dataset = Dataset.from_pandas(test_df)


# from transformers import DistilBertTokenizerFast

# # Instantiate DistilBERT tokenizer...we use the Fast version to optimize runtime
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)


# train_tokenized = train_dataset.map(tokenize_function, batched=True)
# test_tokenized = test_dataset.map(tokenize_function, batched=True)

# print(train_tokenized)

# # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# from transformers import DataCollatorWithPadding

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_tokenized,
#     eval_dataset=test_tokenized,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()





