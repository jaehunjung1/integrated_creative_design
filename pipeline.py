import json
import os
import nltk
import pickle
from tqdm import tqdm
import ipdb

from ner.generate_cloze import prepare, paragraph_to_cloze


origin_dir = os.getcwd()

all_file_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("dataset/wiki_json")) for f in fn]
all_file_list = list(filter(lambda x: 'all' not in x and 'cloze' not in x, all_file_list))

for idx in range(0, len(all_file_list), 100):
    file_list = all_file_list[idx:idx+100]

    os.chdir(os.path.join(origin_dir, 'ner'))
    ner_model = prepare()
    os.chdir('..')

    paragraph_list = []
    paragraph_len_list = []
    answer_list = []
    cloze_list = []
    question_len_list = []
    starting_indices_list = []

    for fname in file_list:
        with open(fname, 'r') as f:
            text_list = [json.loads(s)['text'] for s in f.read().splitlines()]

        for text in text_list:
            sentence_list = text.split('다. ')[1:]  # removes title sentence
            para_in_text = ["다. ".join(sentence_list[i:i+10]) for i in range(0, len(sentence_list), 10)]
            paragraph_len_list.append(len(para_in_text))
            paragraph_list.extend(para_in_text)

    os.chdir('ner')

    # Convert paragraphs to cloze sequences + answers
    for paragraph in tqdm(paragraph_list):
        answers, clozes, starting_indices = paragraph_to_cloze(ner_model, paragraph)
        clozes = [" ".join(nltk.word_tokenize(s)) for s in clozes]

        # filter by cloze sequence length
        length_list = [i for i, cloze in enumerate(clozes) if 10 < len(cloze) < 200]
        length_list = [i for i in length_list if i % 2 == 1]
        answers = [answers[i] for i in length_list]
        clozes = [clozes[i] for i in length_list]
        starting_indices = [starting_indices[i] for i in length_list]

        answer_list.extend(answers)
        cloze_list.extend([s.replace("< ", "<").replace(" >", ">") for s in clozes])
        question_len_list.append(len(answers))
        starting_indices_list.extend(starting_indices)

    os.chdir(origin_dir)

    # Write out the cloze sequences
    os.chdir('UnsupervisedMT/NMT/')
    with open(f"data/clozes/dev{idx//100 + 1}.cl.tok", "w") as f:
        f.write("\n".join(cloze_list))

    os.chdir(os.path.join(origin_dir, 'dataset'))

    # Write out the paragraphs
    with open(f"paragraphs/dev{idx//100 + 1}.paragraph.pkl", "wb") as f:
        pickle.dump(paragraph_list, f)

    # Write out the number of questions per list
    with open(f"paragraphs/dev{idx//100 + 1}.num_question.pkl", "wb") as f:
        pickle.dump(question_len_list, f)

    # Write out the answers
    with open(f"answers/dev{idx//100 + 1}.answer.pkl", "wb") as f:
        pickle.dump(answer_list, f)

    # Write out the answers starting index
    with open(f"answers/dev{idx//100 + 1}.answer_index.pkl", "wb") as f:
        pickle.dump(starting_indices_list, f)

    # Write out the number of paragraphs
    with open(f"paragraphs/dev{idx//100+1}.num_paragraph.pkl", "wb") as f:
        pickle.dump(paragraph_len_list, f)





