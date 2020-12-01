import json
import pickle
import uuid


DATASET_INDEX = list(range(1, 7))
DATASET_INDEX = [1]  # TODO

def filter_mask(sentence):
    return sentence.replace('<인기관명>', '')\
        .replace('<장소>', '')\
        .replace('<기타>', '')\
        .replace('<시간적>', '')\
        .replace('<수량>', '')


for idx in DATASET_INDEX:
    with open(f"dataset/answers/dev{idx}.answer.pkl", 'rb') as f:
        answer_list = pickle.load(f)

    with open(f"dataset/answers/dev{idx}.answer_index.pkl", 'rb') as f:
        answer_index_list = pickle.load(f)

    with open(f"dataset/paragraphs/dev{idx}.num_paragraph.pkl", 'rb') as f:
        num_paragraph_list = pickle.load(f)

    with open(f"dataset/paragraphs/dev{idx}.paragraph.pkl", 'rb') as f:
        paragraph_list = pickle.load(f)

    with open(f"dataset/paragraphs/dev{idx}.num_question.pkl", 'rb') as f:
        num_question_list = pickle.load(f)

    with open(f"UnsupervisedMT/NMT/data/clozes/output{idx}.txt", 'r') as f:
        question_list = f.read().splitlines()
        question_list = [filter_mask(s) for s in question_list]

    qa_index = 0
    paragraph_index = 0

    augmented_dataset = {'data': []}

    # create json file containing everything
    for num_paragraph in num_paragraph_list:
        json_paragraphs = {'title': uuid.uuid1().hex, 'paragraphs': []}

        paragraphs = paragraph_list[paragraph_index:paragraph_index+num_paragraph]
        for paragraph_sub_index, paragraph in enumerate(paragraphs):
            num_qa = num_question_list[paragraph_index + paragraph_sub_index]
            questions = question_list[qa_index:qa_index + num_qa]
            answers = answer_list[qa_index:qa_index + num_qa]
            answer_indices = answer_index_list[qa_index:qa_index + num_qa]

            qa_index += num_qa

            qas = []
            for data_answer, data_answer_index, data_question in zip(answers, answer_indices, questions):
                qas.append({
                    'answers': [{'text': data_answer['word'].strip(),
                                 'answer_start': data_answer_index,
                                 }],
                    'id': uuid.uuid1().hex,
                    'question': data_question
                })

            data = {'context': paragraph, 'qas': qas}
            json_paragraphs['paragraphs'].append(data)

        paragraph_index += num_paragraph

        augmented_dataset['data'].append(json_paragraphs)

    wrong_count = 0
    total_count = 0

    augmented_dataset['data'] = list(filter(lambda x: len(x) > 0, augmented_dataset['data']))
    for data in augmented_dataset['data']:
        data['paragraphs'] = list(filter(lambda x: len(x['qas']) > 0, data['paragraphs']))

        for paragraph in data['paragraphs']:
            start_indices = [qa['answers'][0]['answer_start'] for qa in paragraph['qas']]
            answers = [qa['answers'][0]['text'] for qa in paragraph['qas']]
            paragraph['qas'] = [qa for i, qa in enumerate(paragraph['qas'])
                                if answers[i] in paragraph['context'][start_indices[i]:start_indices[i]+len(answers[i])]]

    for data in augmented_dataset['data']:
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                total_count += 1
                start_index = qa['answers'][0]['answer_start']
                answer = qa['answers'][0]['text']
                if answer != paragraph['context'][start_index:start_index+len(answer)]:
                    wrong_count += 1

    print(total_count)

    with open(f"dataset/KorQuAD/augmented{idx}.json", 'w') as f:
        json.dump(augmented_dataset, f)
