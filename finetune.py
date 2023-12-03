import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import nltk
import numpy as np
from evaluate import load



tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

max_input_length = 1024
max_target_length = 128
def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

metric = load("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_data = load_dataset("samsum")
tokenized_datasets = training_data.map(preprocess_function, batched=True)

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    per_device_train_batch_size=2,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

subset_train_dataset = tokenized_datasets["train"].shuffle(seed=1).select([i for i in range(10)])

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=subset_train_dataset,
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
# model.save_pretrained("./fine_tuned_model")
# tokenizer.save_pretrained("./fine_tuned_tokenizer")