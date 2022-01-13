from textattack.transformations import WordSwapRandomCharacterDeletion, WordSwapQWERTY, CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.augmentation import Augmenter
from textattack.augmentation import EmbeddingAugmenter
import json
import random

def convert_questions(data_path, model, augmenter):
    '''
    returns a list of questions and their ids
    questions are augmented (spelling errors)
    '''
    # val_contexts, val_questions, val_answers, val_ids = read_squad(valid_path)

    # opens the validation file
    with open(data_path, 'rb') as f:
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

    first_n = 5
    # stores the first 100 validation datasets
    val_contexts, val_questions, val_ids = val_contexts[0:first_n], val_questions[0:first_n], val_ids[0:first_n]

    modified_questions = []

    # apply
    for q in val_questions:
        result_q = augmenter.augment(q)
        modified_questions.append(result_q[0])

    return val_ids, modified_questions

def fix_number_bug(string):
    '''
    accepts a string
    fixes textattack bugs
    returns a fixed string
    '''
    result = ''
    string = string.replace('\'', '')
    for i in string:
        if i.isnumeric():
            result += i + ' '
        else:
            result += i
    return result

def create_adversary(data_path, augmenter, question_limit=200000):
    '''
    creates an adversary dataset file from a given squad file
    data_path: string path to the dataset
    question_limit: number  of modified question
    augmenter: creates data augmentation
    '''

    # opens the validation file
    with open(data_path, 'rb') as f:
        squad_dict = json.load(f)


    index = 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                if index == question_limit:
                    return squad_dict
                # augment
                question = qa['question']
                if random.random() > 0.5:
                    try:
                        error_question = augmenter.augment(question)
                        qa['question'] = error_question[0]
                        qa['adversarial'] = 1
                    except IndexError:
                        try:
                            error_question = augmenter.augment(fix_number_bug(question))
                            qa['question'] = error_question[0]
                            qa['adversarial'] = 1
                        except IndexError:
                            print('failed: ', question)
                            qa['adversarial'] = 0
                else:
                    qa['question'] = question
                    qa['adversarial'] = 0

                if index % 1000 == 0:
                    print(index)
                index += 1

    return squad_dict

def write_file(filename, squad_dict):
    '''
    writes the dictionary to a json file
    '''
    # write to file
    json_dataset = json.dumps(squad_dict)
    with open(filename, 'w') as file:
        file.write(json_dataset)



if __name__ == '__main__':
    transformation = CompositeTransformation([WordSwapRandomCharacterDeletion(), WordSwapQWERTY()])
    constraints = [RepeatModification(), StopwordModification()]

    # initiate augmenter
    augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.3,
    transformations_per_example=1
    )

    # additional parameters can be modified if not during initiation
    augmenter.enable_advanced_metrics = True
    augmenter.fast_augment = True
    augmenter.high_yield = True

    # example
    s = 'Who gave their name to Normandy in the century 4'
    results = augmenter.augment(s)

    # read the validation squad files and write predictions to FILE
    FILE = '03validswap.json'
    squad_dict = create_adversary('squad/dev-v2.0.json', augmenter)
    write_file(FILE, squad_dict)

    # augmentations = results[0]
    # perplexity_score = results[1]
    # use_score = results[2]
