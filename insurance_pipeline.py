import os
import re
import hgtk
from tqdm import tqdm
import nltk
import pickle

from ner.generate_cloze import prepare, paragraph_to_cloze

import ipdb

PARAGRAPH_THRESHOLD = 300
FILE_LIST = [f"dataset/insurance/{f}" for f in os.listdir("dataset/insurance") if f.endswith(".txt")]
origin_dir = os.getcwd()


def filter_eomi(text):
    text = text.replace("습니", "")
    jamos = hgtk.text.decompose(text)
    jamos = jamos.replace("ㅂᴥㄴㅣᴥㄷㅏᴥ", "ㄴᴥㄷㅏᴥ")
    text = hgtk.text.compose(jamos)
    return text


paragraph_list = []  # we don't need paragraph_len since all paragraphs are separate.
answer_list = []
cloze_list = []
question_len_list = []
starting_indices_list = []

os.chdir(os.path.join(origin_dir, 'ner'))
ner_model = prepare()
os.chdir(origin_dir)

for fname in tqdm(FILE_LIST):
    temp_paragraph_list = []  # we don't need paragraph_len since all paragraphs are separate.
    temp_answer_list = []
    temp_cloze_list = []
    temp_question_len_list = []
    temp_starting_indices_list = []

    with open(fname, 'r') as f:
        text = filter_eomi(f.read())

    clauses = re.split(r'\n제[1-9]+조.*', text)

    for clause in clauses:
        if len(clause) < PARAGRAPH_THRESHOLD:
            continue
        if len(clause) > 2 * PARAGRAPH_THRESHOLD:
            sentences = clause.split('\n')
            num_sentences_per_paragraph = int(1.5 * len(sentences) / (len(clause) / PARAGRAPH_THRESHOLD))
            sentences = [" ".join(sentences[i:i+num_sentences_per_paragraph])
                         for i in range(0, len(sentences), num_sentences_per_paragraph)]
            temp_paragraph_list.extend(filter(lambda x: len(x) > 200, sentences))
        else:
            clause = clause.replace('\n', ' ')
            temp_paragraph_list.append(clause)

    os.chdir('ner')

    paragraph_to_remove = []

    # Convert paragraphs to cloze sequences + answers
    for paragraph_idx, paragraph in enumerate(temp_paragraph_list):
        answers, clozes, starting_indices = paragraph_to_cloze(ner_model, paragraph)
        clozes = [" ".join(nltk.word_tokenize(s)) for s in clozes]

        # filter by cloze sequence length
        length_list = [i for i, cloze in enumerate(clozes) if 10 < len(cloze) < 200]
        length_list = [i for i in length_list if i % 2 == 1]
        answers = [answers[i] for i in length_list]
        clozes = [clozes[i] for i in length_list]
        starting_indices = [starting_indices[i] for i in length_list]

        if len(answers) == 0:
            paragraph_to_remove.append(paragraph_idx)
            continue

        temp_answer_list.extend(answers)
        temp_cloze_list.extend([s.replace("< ", "<").replace(" >", ">") for s in clozes])
        temp_question_len_list.append(len(answers))
        temp_starting_indices_list.extend(starting_indices)

    temp_paragraph_list = [s for i, s in enumerate(temp_paragraph_list) if i not in paragraph_to_remove]

    paragraph_list.extend(temp_paragraph_list)
    answer_list.extend(temp_answer_list)
    cloze_list.extend(temp_cloze_list)
    question_len_list.extend(temp_question_len_list)
    starting_indices_list.extend(temp_starting_indices_list)

    os.chdir(origin_dir)

# Write out the cloze sequences
os.chdir(os.path.join(origin_dir, 'UnsupervisedMT/NMT/'))
with open(f"data/clozes/dev_insurance.cl.tok", "w") as f:
    f.write("\n".join(cloze_list))

os.chdir(os.path.join(origin_dir, 'dataset'))

# Write out the paragraphs
with open(f"paragraphs/dev_insurance.paragraph.pkl", "wb") as f:
    pickle.dump(paragraph_list, f)

# Write out the number of questions per list
with open(f"paragraphs/dev_insurance.num_question.pkl", "wb") as f:
    pickle.dump(question_len_list, f)

# Write out the answers
with open(f"answers/dev_insurance.answer.pkl", "wb") as f:
    pickle.dump(answer_list, f)

# Write out the answers starting index
with open(f"answers/dev_insurance.answer_index.pkl", "wb") as f:
    pickle.dump(starting_indices_list, f)

print(f"# of Paragraphs: {len(paragraph_list)}, # of Q-A pairs: {len(answer_list)}")
