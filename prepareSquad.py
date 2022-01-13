import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer, ElectraForQuestionAnswering, ElectraConfig, ElectraTokenizerFast
import json
from pathlib import Path

# https://huggingface.co/transformers/custom_datasets.html#qa-squad
def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    contexts = []
    questions = []
    answers = []
    ids = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                # if qa['adversarial'] == 0:
                #     continue
                question = qa['question']
                id = qa['id']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    ids.append(id)

    return contexts, questions, answers, ids

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def prepareData(trainpath, validpath):
    '''
    Make sure data is downloaded
    :::::
    mkdir squad
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad/train-v2.0.json
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad/dev-v2.0.json
    :::::
    returns train_dataset, val_dataset
    '''
    # split dataset
    train_contexts, train_questions, train_answers, _ = read_squad(trainpath)
    val_contexts, val_questions, val_answers, _ = read_squad(validpath)

    train_contexts, train_questions, train_answers = train_contexts[0:1], train_questions[0:1], train_answers[0:1]
    val_contexts, val_questions, val_answers = val_contexts[0:1], val_questions[0:1], val_answers[0:1]

    print('adding index')
    # fix squad offset
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    print('tokenizing')
    # tokenizer
    tokenizer = ElectraTokenizerFast.from_pretrained('deepset/electra-base-squad2')
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    print('adding token positions')
    # convert character positions to token positions
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)


    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    return train_dataset, val_dataset
