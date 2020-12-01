from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pickle
import torch
from gluonnlp.data import SentencepieceTokenizer
from pathlib import Path

from ner.model.net import KobertCRF
from ner.data_utils.utils import Config
from ner.data_utils.vocab_tokenizer import Tokenizer
from ner.data_utils.pad_sequence import keras_pad_fn

import ipdb


class DecoderFromNamedEntitySequence:
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        # For unbatched
        # input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        # pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # For batched
        input_token = self.tokenizer.decode_token_ids([list_of_input_ids])[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids]

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append(
                        {"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-" + entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": entity_tag, "prob": None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""

        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False
        list_of_start_index = []
        for i, (token_str, pred_ner_tag_str) in enumerate(zip(input_token, pred_ner_tag)):
            if i == 0 or i == len(pred_ner_tag) - 1:  # remove [CLS], [SEP]
                continue
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                list_of_start_index.append(len(decoding_ner_sentence))

            decoding_ner_sentence += token_str
            # if 'B-' in pred_ner_tag_str:
            #     if is_prev_entity is True:
            #         decoding_ner_sentence += ':' + prev_entity_tag+ '>'
            #
            #     if token_str[0] == ' ':
            #         token_str = list(token_str)
            #         token_str[0] = ' <'
            #         token_str = ''.join(token_str)
            #         decoding_ner_sentence += token_str
            #     else:
            #         decoding_ner_sentence += '<' + token_str
            #     is_prev_entity = True
            #     prev_entity_tag = pred_ner_tag_str[-3:]  # 첫번째 예측을 기준으로 하겠음
            #     is_there_B_before_I = True
            #
            # elif 'I-' in pred_ner_tag_str:
            #     decoding_ner_sentence += token_str
            #
            #     if is_there_B_before_I is True:  # I가 나오기전에 B가 있어야하도록 체크
            #         is_prev_entity = True
            # else:
            #     if is_prev_entity is True:
            #         decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
            #         is_prev_entity = False
            #         is_there_B_before_I = False
            #     else:
            #         decoding_ner_sentence += token_str
        return list_of_ner_word, decoding_ner_sentence, list_of_start_index


def prepare():
    model_dir = Path('./experiments/base_model_with_crf')
    model_config = Config(json_path=model_dir / 'config.json')

    # load vocab & tokenizer
    tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = model.state_dict()
    checkpoint = torch.load("./experiments/base_model_with_crf/best-epoch-16-step-1500-acc-0.993.bin",
                            map_location=torch.device('cpu'))
    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v
    model.load_state_dict(convert_keys)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)
    return tokenizer, device, model, decoder_from_res


def paragraph_to_cloze(prepared_inputs, paragraph: str):
    def pad(num_list, pad_idx):
        lens = [len(x) for x in num_list]
        max_len = max(lens)
        return torch.tensor([x + [pad_idx] * (max_len - len(x)) for x in num_list]).long(), lens

    tokenizer, device, model, decoder_from_res = prepared_inputs

    # no batch
    # list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([paragraph])
    # x_input = torch.tensor(list_of_input_ids).long().to(device)
    # list_of_pred_ids = model(x_input)
    #
    # list_of_cloze = []
    #
    # list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids,
    #                                                            list_of_pred_ids=list_of_pred_ids)
    #
    # for ner_word in list_of_ner_word:
    #     cloze = decoding_ner_sentence.replace(ner_word['word'], get_mask(ner_word['tag']))
    #     list_of_cloze.append(cloze)

    # batched operations
    unstripped_sentences = list(map(lambda x: x + ".", paragraph.split(".")))[:-1][:100]
    lstripped_sentences = list(map(lambda x: x.lstrip() + ".", paragraph.split(".")))[:-1][:100]
    sentences = list(map(lambda x: x.strip() + ".", paragraph.split(".")))[:-1][:100]  # just in case for OOM

    lists_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids(sentences)
    lists_of_input_ids = list(filter(lambda x: len(x) < 500, lists_of_input_ids))

    if len(lists_of_input_ids) == 0:
        return [], [], []

    input_tensor, lens = pad(lists_of_input_ids, tokenizer._vocab.PAD_ID)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        lists_of_pred_ids = model(input_tensor)
    lists_of_pred_ids = [lists_of_pred_ids[i][:length] for i, length in enumerate(lens)]

    lists_of_ner_word = []
    lists_of_cloze = []
    lists_of_starting_index = []
    for i, (list_of_input_ids, list_of_pred_ids) in enumerate(zip(lists_of_input_ids, lists_of_pred_ids)):
        list_of_ner_word, decoding_ner_sentence, list_of_starting_index = decoder_from_res(
            list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)

        for j, ner_word in enumerate(list_of_ner_word):
            cloze = decoding_ner_sentence.replace(ner_word['word'], get_mask(ner_word['tag']))
            lists_of_cloze.append(cloze)

            starting_index = paragraph.find(unstripped_sentences[i]) + list_of_starting_index[j] \
                             + len(unstripped_sentences[i]) - len(lstripped_sentences[i])
            lists_of_starting_index.append(starting_index)

        lists_of_ner_word.extend(list_of_ner_word)
    return lists_of_ner_word, lists_of_cloze, lists_of_starting_index


def get_mask(entity_tag):
    if entity_tag in ['PER', 'ORG']:
        return ' <인기관명>'
    elif entity_tag in ['LOC']:
        return ' <장소>'
    elif entity_tag in ['POH']:
        return ' <기타>'
    elif entity_tag in ['DAT', 'TIM', 'DUR']:
        return ' <시간>'
    else:
        return ' <수량>'


if __name__ == "__main__":
    paragraph = "보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약의 내용을 이해하고 이행할 능력이 있는 자로서 보험업법 제2조(정의), " \
                "보험업법시행령 제6조의2(전문보험계약자의 범위 등) 또는 보험업감독규정 제1-4조의2(전문보험계약자의 범위)에서 정한 국가, 한국은행, " \
                "대통령령으로 정하는 금융기관, 주권상장법인, 지방자치단체, 단체보험계약자 등의 전문보험계약자를 말합니다. ② 제1항에도 불구하고 " \
                "청약한 날부터 30일이 초과된 계약은 청약을 철회할 수 없습니다. ③ 계약자는 청약서의 청약철회란을 작성하여 회사에 제출하거나, " \
                "통신수단을 이용하여 제1항 의 청약 철회를 신청할 수 있습니다."

    prepared_inputs = prepare()
    entities, clozes, starting_indices = paragraph_to_cloze(prepared_inputs, paragraph)
    ipdb.set_trace()
