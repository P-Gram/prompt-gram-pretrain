
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from data_collator import DataCollatorForWholeWordMask
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from run_mlm_wwm import add_chinese_references
datasets = load_dataset('text', data_files="./data/corpus.txt")

tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }
tokenizer = AutoTokenizer.from_pretrained("/Users/qiwang/company/Document/pretrain-models/bert-base-chinese", **tokenizer_kwargs)

def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=None)


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
tokenized_datasets["train"] = add_chinese_references(tokenized_datasets["train"], "./data/ref.txt")
data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.25, pao_length=12, prompt_length=8)

for data in tqdm(tokenized_datasets["train"]):
    input_ids = data['input_ids']
    input_tokens = [tokenizer._convert_id_to_token(id) for id in input_ids]
    result = data_collator.torch_call([data])
    result['input_ids'] = result['input_ids'].numpy().tolist()
    result['labels'] = result['labels'].numpy().tolist()
    input_mask_tokens = [tokenizer._convert_id_to_token(id) for id in result['input_ids'][0]]
    labels = [tokenizer._convert_id_to_token(id) if id != -100 else '' for id in result['labels'][0]]
    print(input_tokens)
    print(input_mask_tokens)
    print(labels)
    print('-----------------------------------------------------------------------------------------------')