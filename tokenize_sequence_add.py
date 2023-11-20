import os
from transformers import AutoTokenizer
from tqdm import tqdm
from common_utils import clean_text


def data_to_token_classification(filenames, tokenizer, save_filename, reverse=False, single_size=100000):
    dir_name = os.path.dirname(save_filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    index = 0
    f_save = open(f"{save_filename}", "w", encoding='utf-8')
    total_count = 0
    filter = 0
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                items = line.split('\t')
                if len(items) != 3 or len(items[1]) == 0:
                    continue
                items = [x.strip() for x in items]
                # if len(items[1]) != len(items[2]):
                #     print(
                #         'data mismatch, ignore! {}->{}'.format(items[1], items[2]))
                total_count += 1
                if not reverse:
                    pair = items[1:]
                else:
                    pair = items[:0:-1]
                tokens, labels = get_token_labels(tokenizer, clean_text(pair[0]), clean_text(pair[1]))
                temp = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False)
                # 过滤掉会二次subword情况
                if len(temp["input_ids"]) != len(tokens):
                    filter += 1
                    continue
                for token, label in zip(tokens, labels):
                    f_save.write('{}\t{}\n'.format(token, label))
                f_save.write('\n')
                if (i + 1) % single_size == 0:
                    f_save.close()
                    print(f"save file {save_filename}_{index}.txt successfully")
                    index += 1
                    f_save = open(f"{save_filename}_{index}.txt", "w", encoding='utf-8')
    f_save.close()
    print(f"filter num = {filter}")
    return total_count


def get_token_labels(tokenizer, input_sent, output_sent, max_length=512):
    tokenize_result = tokenizer(
        input_sent, return_offsets_mapping=True, truncation=True, max_length=max_length)
    tokens = []
    labels = []
    wrong_sen=input_sent
    right_sen=output_sent
    for offsets in tokenize_result['offset_mapping']:
        if offsets[1] <= offsets[0]:
            continue
        input_token = input_sent[offsets[0]:offsets[1]]
        sub_sent=output_sent[offsets[0]:]
        if sub_sent.startswith('ADD'):
            output_token=output_sent[offsets[0]:offsets[1]+3]
            output_sent=output_sent[:offsets[0]]+output_sent[offsets[1]+3-1:]
        else:
            output_token = output_sent[offsets[0]:offsets[1]]
        labels.append(output_token)
        tokens.append(input_token)
    return tokens, labels


filemaps = {
    "sim": [
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law.train",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law_js.train",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law.trainsep",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law.traintgt",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law.dev",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law_js.dev",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law.devsep",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law.devtgt",
        "/data1/ssq/experiment/ECSpell-main/csc_evaluation/builds/sim/domain/law_js.test",
    ],
}
reverse = False
model_list = ["glyce"]

for model_name in model_list:
    tokenizer = AutoTokenizer.from_pretrained("/data1/ssq/experiment/RSpell/Transformers/glyce")
    for font_type, filenames in filemaps.items():
        save_dir = f'Data/traintest/{font_type}/{model_name}'
        print(f"Model name: {model_name}\tFont type: {font_type}")
        for filename in filenames:
            corpus_type = filename.split("/")[-2]
            print(f'Handle file: {filename}')
            total_count = data_to_token_classification(
                [filename], tokenizer, os.path.join(save_dir, corpus_type, os.path.basename(filename)), reverse=reverse)
            print(f'total count: {total_count}')

