from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')
from rouge.rouge import Rouge
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer




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

    for i in range(len(reference)):
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


        model_name = "gpt2"  # You can specify other GPT-2 model variations as needed
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        input_ids = tokenizer.encode(generated[i], return_tensors="pt")

        # Calculate the negative perplexity
        with torch.no_grad():
            output = model(input_ids, return_dict=True)
            neg_ppl = torch.exp(output.logits).mean()

        neg_ppl_sum += neg_ppl

        if print_all:
            print("Negative PPL:", neg_ppl.item())
    
    if len(reference) > 0:
        print("mean BLEU:", bleu_sum/len(reference))
        print("mean ROUGE_L:", rouge_l_sum/len(reference))
        print("mean ROUGE_2:", rouge_2_sum/len(reference))
        print("mean NEG_PPL:", neg_ppl_sum/len(reference))

    else:
        print("Reference/generated appear to be of length 0")

if __name__ == '__main__':
    with open("reference.txt", "r") as reference_file:
        reference = [reference_file.read()]

    with open("generated.txt", "r") as generated_file:
        generated = [generated_file.read()]

    evaluate(reference, generated, False)