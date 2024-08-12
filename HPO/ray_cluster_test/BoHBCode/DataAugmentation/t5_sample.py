
from cgitb import small
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")


def paraphrase_dataset(dataset):
    paraphrased_dataset = []
    for example in dataset:
        original_text = example["text"]  # Modify this according to your dataset structure
        input_text = "paraphrase: " + original_text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=3, early_stopping=True)
        paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        paraphrased_example = {"original_text": original_text, "paraphrases": paraphrases}
        paraphrased_dataset.append(paraphrased_example)
    return paraphrased_dataset



if __name__ == '__main__':
    dataset = load_dataset("gnad10", split="train")
    small_dataset = dataset.select(range(10))
   
