# compares two predictions side by side
import json
from pathlib import Path

'''
read prediction file as Dictionary
'''
def read_json(file_path):

    # open json
    with open(file_path, 'rb') as f:
        dict = json.load(f)

    return dict

# read squad but used for side by side comparison
def read_squad_side_by_side(path):
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
                answer = qa['answers']
                contexts.append(context)
                questions.append(question)
                answers.append(answer)
                ids.append(id)

    return contexts, questions, answers, ids

# write to file functionn
def write_to_file(adversarial_path, regular_path):
    contexts, questions, answers, ids = read_squad_side_by_side('squad/dev-v2.0.json')
    _, a_questions, _, _ = read_squad_side_by_side('adversarial/valid.json')

    pred_dict_regular = read_json(regular_path)
    pred_dict_adversarial = read_json(adversarial_path)

    f = open('side_by_side_adversarial_data.txt', 'w')
    count = 0
    for index, id in enumerate(ids):
        if id in pred_dict_regular:
            count+=1
            line0 = 'id: ' + id + '\n'
            line1 = 'question: ' + questions[index] + '\n'
            linex = 'modified question: ' + a_questions[index] + '\n'
            line2 = 'context: ' + contexts[index] + '\n'
            line3 = 'actual answer(s): ' + ', '.join(e['text'] for e in answers[index]) + '\n'
            line4 = 'deepset model squad answer: ' + pred_dict_regular[id] + '\n'
            line5 = 'adversarial model answer: ' + pred_dict_adversarial[id] + '\n'

            # determine if answers are different:
            if pred_dict_regular[id] == pred_dict_adversarial[id]:
                line6 = 'same answer\n'
            else:
                line6 = 'different answer\n'

            line = line0 + line1 + linex + line2 + line3 + line4 + line5 + line6 + '\n'
            f.write(line)
    print(count)
    f.close()

if __name__ == '__main__':
    write_to_file('predictions/predAdvAdvModel.json', 'predictions/predAdversarial.json')
