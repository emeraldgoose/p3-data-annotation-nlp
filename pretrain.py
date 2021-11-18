import gc

import torch
from transformers import Trainer, TrainingArguments, BertConfig, BertTokenizer, BertForMaskedLM

from tokenizers.implementations import BertWordPieceTokenizer


class MLM_dataset(torch.utils.data.Dataset):
    """ masked learning dataset """

    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.dataset.items()}
        item['labels'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)


def tokenizer_train():
    # load sentences
    tokenizer = BertWordPieceTokenizer(
        vocab=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##",
    )

    limit_alphabet = 1000
    vocab_size = 30000

    tokenizer.train(
        files='./sentence.txt',
        vocab_size=vocab_size,
        limit_alphabet=limit_alphabet,
    )

    tokenizer.save("./tokenizer.json", True)  # save tokenizer.json
    tokenizer.save_model('./')  # save vocab.txt


def preprocessing(dataset):
    """ preprocessing(random word convert to "[MASK]") """

    mask = 4  # [MASK]
    label = []
    for idx, words in enumerate(dataset['input_ids']):
        maxlen = 0
        for i in range(len(words)):
            if not (0 <= words[i] <= 4):
                maxlen = max(maxlen, i)

        masked_idx = torch.randint(size=(3,), low=1, high=maxlen)
        tmp = words.clone().detach()
        label.append(tmp)

        for index in masked_idx:
            words[index] = mask
        dataset['input_ids'][idx] = words

    return dataset, label


def tokenized_dataset(dataset, tokenizer):
    tokenized_sentence = tokenizer(
        dataset,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=200,
        add_special_tokens=True,
    )
    return tokenized_sentence


def make_dataset(tokenizer):
    # make dataset
    with open('sentence.txt', 'r', encoding='utf-8') as f:
        dataset = f.read()
        dataset = dataset.split('\n')
        dataset = [line.strip() for line in dataset]

    dataset = tokenized_dataset(dataset, tokenizer)

    dataset, label = preprocessing(dataset)

    dataset = MLM_dataset(dataset, label)

    return dataset


def train(dataset, model, tokenizer, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir='./models')
    trainer.save_state()


def main():
    """ Bert base pretraining """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)

    # run garbage collect, empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # train tokenizer and make user vocab.txt
    # tokenizer_train()

    # # training 준비
    config = BertConfig(
        vocab_size=7243,
        max_position_embeddings=512,
        architectures=["BertForMaskedLM"]
    )

    # config.save_pretrained('./')  # save config.json

    tokenizer = BertTokenizer(vocab_file='./vocab.txt', do_lower_case=False)
    tokenizer.model_max_length = 512

    # tokenizer.save_pretrained('./')  # save tokenizer_config.json

    model = BertForMaskedLM(config=config)
    model.resize_token_embeddings(config.vocab_size)
    model.to(device)

    training_args = TrainingArguments(
        num_train_epochs=3,
        do_train=True,
        output_dir='./results/',
        learning_rate=5e-5,
        logging_steps=1,
        per_device_train_batch_size=8,
    )

    # make datasets
    dataset = make_dataset(tokenizer)

    # train
    train(dataset=dataset, model=model,
          tokenizer=tokenizer, training_args=training_args)


if __name__ == "__main__":
    main()
