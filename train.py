import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import ElectraTokenizer, ElectraForQuestionAnswering, ElectraConfig, pipeline
from prepareSquad import *
from eval import *
import time

# https://huggingface.co/transformers/model_doc/electra.html#electraforquestionanswering

def save_model(model, epoch):
    from torch import save
    from os import path
    path = 'models/' + str(epoch) + 'adv.th'
    return save(model.state_dict(), path)

def load_model(name):
    from torch import load
    from os import path
    r = ElectraForQuestionAnswering.from_pretrained('deepset/electra-base-squad2')
    r.load_state_dict(load('models/' + name, map_location='cpu'))
    return r

def finetune(train_dataset):
    '''
    finetunes the electra model on train_dataset
    returns the finetuned model
    '''
    # model
    model = ElectraForQuestionAnswering.from_pretrained('deepset/electra-base-squad2')

    load = True
    if load:
        model = load_model('0adv.th')
        model.eval()

    this_model = True
    if this_model and load:
        return model

    # to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        print('epoch ', epoch)
        loss_list = []

        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss_list.append(loss.item())
            loss.backward()
            optim.step()

        print(sum(loss_list)/len(loss_list))
        save_model(model, epoch)

    model.eval()
    return model

def create_prediction_file(valid_path, model):
    '''
    creates pred.json which contains predictions for question id and answer
    valid_path is the path to the validation dataset
    model is the electra finetuned model which would evaluate our results
    '''
    # val_contexts, val_questions, val_answers, val_ids = read_squad(valid_path)
    start = time.time()

    # opens the validation file
    with open(valid_path, 'rb') as f:
        squad_dict = json.load(f)

    # reads the validation file
    val_contexts, val_questions, val_ids = [], [], []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                val_questions.append(qa['question'])
                val_ids.append(qa['id'])
                val_contexts.append(context)

    # start position and length of the number of questions
    START = 1000
    LIMIT = 1500
    END = LIMIT + START
    # stores LIMIT validation datasets starting from START
    val_contexts, val_questions, val_ids = val_contexts[START:END], val_questions[START:END], val_ids[START:END]

    # prediction in the form {id:answer}
    predict = {}

    # choose model (electra based squad or your own model)
    model_name = "deepset/electra-base-squad2"

    # predict using pipeline
    nlp = pipeline('question-answering', model=model, tokenizer=model_name)
    for i in range(len(val_questions)):
        QA_input = {
            'question': val_questions[i],
            'context': val_contexts[i]
        }
        id = val_ids[i]
        res = nlp(QA_input)
        print(res)
        predict[id] = res['answer']

    # write to file
    json_predict = json.dumps(predict)
    FILENAME = '03validswap_regular.json'

    with open(FILENAME, 'w') as file:
        file.write(json_predict)

    end = time.time()
    print(f'Duration: {end - start} seconds')

# https://huggingface.co/deepset/electra-base-squad2
if __name__ == '__main__':

    # prediction from file FILENAME
    FILENAME = 'adversarial/03validswap.json'

    # creates prediction file using original model
    model_name = "deepset/electra-base-squad2"
    create_prediction_file(FILENAME, model_name)

    # adding stopping point if you don't want to continue
    # exit()

    # fine tune the deepset model
    train_dataset, valid_dataset = prepareData('adversarial/train.json', 'adversarial/valid.json')
    print('example:', train_dataset[0])
    model = finetune(train_dataset)

    # create prediction file using trained model
    create_prediction_file(FILENAME, model=model)

    # example of pipeline answering question
    nlp = pipeline('question-answering', model=model, tokenizer=model_name)
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    print(res)
