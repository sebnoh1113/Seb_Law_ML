# import pickle
# import pandas as pd
# import numpy as np

# from tqdm import tqdm

# from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# from kobert_tokenizer import KoBERTTokenizer
# from transformers import BertForSequenceClassification
# from transformers import AdamW
# from transformers.optimization import get_cosine_schedule_with_warmup

## 우선 딥러닝 자연어처리 모델의 활용을 위해서 tokenizer/model의 획득 및 점검이 선행되어야 함
from transformers import PreTrainedTokenizerFast
# from transformers.utils.dummy_pt_objects import PegasusPreTrainedModel

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>',
                                                    eos_token = '</s>',
                                                    unk_token='<unk>',
                                                    pad_token = '<pad>',
                                                    mask_token = "<mask>")
tokenizer.tokenize("안녕하세요. 한국어 GPT-2입니다.") 

import pickle

with open('kogpt2basev2tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)
with open('kogpt2basev2tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2") 
with open('kogpt2basev2model.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
with open('kogpt2basev2model.pickle', 'rb') as f:
    model = pickle.load(f)
    
text = "돈을 많이 벌려면 어떤 투자를 해야 하는지"
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids]),
                         max_length = 128,
                         repetition_penalty = 2.0,
                         pad_token_id = tokenizer.pad_token_id,
                         eos_token_id = tokenizer.eos_token_id,
                         bos_token_id = tokenizer.bos_token_id,
                         use_cache = True)
generated = tokenizer.decode(gen_ids[0, :])
print(generated)