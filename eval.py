from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')
from rouge.rouge import Rouge
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
import sys
import os

@contextmanager
def suppress_output():
    # Save the original stdout
    original_stdout = sys.stdout
    
    # Redirect stdout to a null file
    with open(os.devnull, 'w') as null_file:
        sys.stdout = null_file
        yield
    
    # Restore the original stdout
    sys.stdout = original_stdout


def evaluate(reference: list, generated: list, print_all=False):
    """
    reference is a list of gold label text,
    generated is a corresponding list of the generated text, each index should match the same index on reference,
    print_all: should the function print the results for each item in list
    """
    bleu_sum = 0
    rouge_l_sum = 0
    rouge_2_sum = 0
    neg_ppl_sum = 0

    assert len(reference) == len(generated)

    model_name = "gpt2"  # You can specify other GPT-2 model variations as needed
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    for i in range(len(reference)):
        if (i % 100 == 0) and (i != 0):
            print(f"{i}/{len(reference)}, mean BLEU: {bleu_sum/i}, mean ROUGE_L: {rouge_l_sum/i}, mean ROUGE_2: {rouge_2_sum/i}, mean NEG_PPL: {neg_ppl_sum/i}")
 
        # Calculate BLEU score
        reference_tokens = reference[i].split()
        generated_tokens = generated[i].split()

        bleu_score = sentence_bleu([reference_tokens], generated_tokens)
        bleu_sum += bleu_score
        if print_all:
            print("BLEU Score:", bleu_score) # this cannot be compared to the one generated in the paper, the bleu score implementation they use required tensorflow 1.x
        

        # Calculate ROUGE-2 and ROUGE-L scores
        rouge = Rouge(metrics=['rouge-n', 'rouge-l'],
                                max_n=2,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                alpha=0.5, # Default F1_score
                                weight_factor=1.2,
                                stemming=True)
        rouge_scores = rouge.get_scores([generated[i]], [reference[i]])
        # print(rouge_scores)
        rouge_2_score = rouge_scores['rouge-2']['f']
        rouge_l_score = rouge_scores['rouge-l']['f']

        rouge_l_sum += rouge_l_score
        rouge_2_sum += rouge_2_score

        if print_all:
            print("ROUGE-2 Score:", rouge_2_score)
            print("ROUGE-L Score:", rouge_l_score)



        encodings = tokenizer("\n\n".join(generated[i]), return_tensors="pt")

        max_length = model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride), disable=True):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

            ppl = torch.exp(torch.stack(nlls).mean())
            neg_ppl = -torch.log(ppl)

        neg_ppl_sum += neg_ppl.item()  # accumulate the negative perplexity

        if print_all:
            print("Negative PPL:", neg_ppl.item(), "\n")
    
    if len(reference) > 0:
        print("mean BLEU:", bleu_sum/len(reference))
        print("mean ROUGE_L:", rouge_l_sum/len(reference))
        print("mean ROUGE_2:", rouge_2_sum/len(reference))
        print("mean NEG_PPL:", neg_ppl_sum/len(reference))

    else:
        print("Reference/generated appear to be of length 0")

if __name__ == '__main__':
    df = pd.read_csv("finetuned_flan_0.001_15.csv")
    reference = df["label"].tolist()
    generated = df["output"].tolist()

    # reference = ["A quick brown fox jumps over a lazy dog"]
    # generated = ["The quick brown fox jumps over the lazy dog"]

    evaluate(reference, generated, True)