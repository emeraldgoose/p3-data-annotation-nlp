import gc

import torch
from transformers import Trainer, TrainingArguments, BertConfig, BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoConfig

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

        masked_idx = torch.randint(
            size=(int((maxlen-1)*0.15*0.8),), low=1, high=maxlen)
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
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentence


def make_dataset(tokenizer):
    # make dataset
    with open('sentence.txt', 'r', encoding='utf-8') as f:
        dataset = f.read()
        dataset = dataset.split('\n')
        dataset = [line.strip() for line in dataset]
        dataset += dataset
        dataset += dataset

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
    # trainer.save_state()


def main():
    """ Bert base pretraining """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # run garbage collect, empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # train tokenizer and make user vocab.txt
    # tokenizer_train()

    # # training 준비
    config = BertConfig.from_pretrained('emeraldgoose/bert-base-v1-sports')

    # config.save_pretrained('./')  # save config.json

    tokenizer = AutoTokenizer.from_pretrained(
        'emeraldgoose/bert-base-v1-sports')

    # tokenizer.save_pretrained('./')  # save tokenizer_config.json

    model = BertForMaskedLM(config=config)
    model.resize_token_embeddings(config.vocab_size)
    model.to(device)

    training_args = TrainingArguments(
        num_train_epochs=80,
        do_train=True,
        output_dir='./results/',
        learning_rate=5e-5,
        logging_steps=10,
        per_device_train_batch_size=128,
    )

    # make datasets
    dataset = make_dataset(tokenizer)
    print(f'data length = {len(dataset)}')
    # train
    train(dataset=dataset, model=model,
          tokenizer=tokenizer, training_args=training_args)


def evalution():
    tokenizer = AutoTokenizer.from_pretrained('./models')
    config = AutoConfig.from_pretrained('./models')
    model = BertForMaskedLM.from_pretrained('./models', config=config)

    text = "산악 자전거 경기는 상대적으로 [MASK] 경기로 1990년대에 활성화 되었다."
    print([tokenizer.decode(token) for token in tokenizer.encode(text)])
    inputs = tokenizer.encode(text, return_tensors='pt')

    model.eval()
    outputs = model(inputs)['logits']
    predict = outputs.argmax(-1)
    print([tokenizer.decode(iter) for iter in predict])


if __name__ == "__main__":
    main()
    evalution()
