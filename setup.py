import numpy as np
import os, re, json, torch
from datasets import load_dataset
from transformers import BertTokenizerFast



def tokenize_data(data_obj, tokenizer):
    tokenized = []
    for elem in data_obj:
        tokenized.append({'hist': tokenizer(elem['hist']).input_ids,
                          'uttr': tokenizer(elem['uttr']).input_ids,
                          'resp': tokenizer(elem['resp']).input_ids})       
    return tokenized



def process_data(orig_data, tokenizer):
    single, double, triple = [], [], []

    for dial in orig_data:
        dial_list = []
        dial_turns = len(dial)
        
        for uttr in dial:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr)
            dial_list.append(_uttr.strip().lower())

        if len(single) < 10000:
            single.append({'hist': dial_list[0],
                            'uttr': dial_list[0], 
                            'resp': dial_list[1]})
        
        if len(double) < 13000:
            if dial_turns > 4:
                diff = dial_turns - 4
                if not diff:
                    double.append({'hist': dial_list[-4:-2], 
                                    'uttr': dial_list[-2], 
                                    'resp': dial_list[-1]})
                else:
                    for i in range(1, diff+1):
                        double.append({'hist': dial_list[-4-i:-2-i], 
                                        'uttr': dial_list[-2-i], 
                                        'resp': dial_list[-1-i]})

        if len(triple) < 13000:
            if dial_turns > 6:
                diff = dial_turns - 6
                if not diff:
                    triple.append({'hist': dial_list[-6:-2], 
                                    'uttr': dial_list[-2], 
                                    'resp': dial_list[-1]})
                else:
                    for i in range(1, diff+1):
                        triple.append({'hist': dial_list[-6-i:-2-i], 
                                        'uttr': dial_list[-2-i], 
                                        'resp': dial_list[-1-i]})
                        
        if len(single) >= 10000 and len(double) >= 13000 and len(triple) >= 13000:
            break

    train = tokenize_data(single[:10000] + double[:10000] + triple[:10000], tokenizer)
    valid = tokenize_data(double[10000:11500] + triple[10000:11500], tokenizer)
    test = tokenize_data(double[11500:13000] + triple[11500:13000], tokenizer)
    
    return train, valid, test




def save_data(data_obj, split):
    with open(f'data/{split}_2nd.json', 'w') as f:
        json.dump(data_obj, f)                   



def main():
    orig = load_dataset('daily_dialog', split='train')['dialog']
    tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-small')
    train, valid, test = process_data(orig, tokenizer)
    
    save_data(train, 'train')
    save_data(valid, 'valid')
    save_data(test, 'test')



if __name__ == '__main__':
    main()