import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import pad_sequence

import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
# model = AutoModel.from_pretrained("skt/kobert-base-v1")
# result = kobert_tokenizer.tokenize("너는 내년 대선 때 투표할 수 있어?")
# print(result)
# kobert_vocab = kobert_tokenizer.get_vocab()
# print(kobert_vocab.get('▁대선'))
# print([kobert_tokenizer.encode(token) for token in result])

# from transformers import AutoTokenizer, AutoModelForMaskedLM
# kcbert의 tokenizer와 모델을 불러옴.
# kcbert_tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
# kcbert = AutoModelForMaskedLM.from_pretrained("beomi/kcbert-base")
# result = kcbert_tokenizer.tokenize("너는 내년 대선 때 투표할 수 있어?")
# print(result)
# print(kcbert_tokenizer.vocab['대선'])
# print([kcbert_tokenizer.encode(token) for token in result])

dataFileName = './dfFinal.csv'
x_dataFieldName = 'precSentences'
y_dataFieldName = 'case_sort'

sampleRatio = 0.03 # 실전의 경우 1
targetDimension = 512 # pretrained model에 따라 조정
num_labels = 7 # transfer learing 데이터에 따라 조정
dr_rate = 0.4
# device = torch.device("cpu") # model / dataset ...to(device)
    
batch_size = 2
epochs = 4 

# AdamW 사용시 필요
warmup_steps = None
warmup_ratio = 0.1
weight_decay = 0.01

max_grad_norm = 1
log_interval = 100
learning_rate = 5e-5

class Vocabulary(object):
    
    """매핑을 위해 텍스트를 처리하고 어휘 사전을 만드는 클래스 """

    def __init__(self, token_to_idx=None):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
    # def to_serializable(self):
        # """ 직렬화할 수 있는 딕셔너리를 반환합니다 """
        # return {'token_to_idx': self._token_to_idx}

    # @classmethod
    # def from_serializable(cls, contents):
        # """ 직렬화된 딕셔너리에서 Vocabulary 객체를 만듭니다 """
        # return cls(**contents)

    def add_token(self, token):
        """ 토큰을 기반으로 매핑 딕셔너리를 업데이트합니다

        매개변수:
            token (str): Vocabulary에 추가할 토큰
        반환값:
            index (int): 토큰에 상응하는 정수
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
            
    def add_many(self, tokens):
        """토큰 리스트를 Vocabulary에 추가합니다.
        
        매개변수:
            tokens (list): 문자열 토큰 리스트
        반환값:
            indices (list): 토큰 리스트에 상응되는 인덱스 리스트
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """토큰에 대응하는 인덱스를 추출합니다.
        
        매개변수:
            token (str): 찾을 토큰 
        반환값:
            index (int): 토큰에 해당하는 인덱스
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """ 인덱스에 해당하는 토큰을 반환합니다.
        
        매개변수: 
            index (int): 찾을 인덱스
        반환값:
            token (str): 인텍스에 해당하는 토큰
        에러:
            KeyError: 인덱스가 Vocabulary에 없을 때 발생합니다.
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

def dataloader_factory(df, x_dataFieldName, targetDimension, batch_size):       
    # vectorizing
    print("vectorizing...")
    data = {'input_ids':[], 'attention_mask': [], 'token_type_ids': [], 'label': []}
    
    for i in tqdm(range(len(df))):
        tuple = vectorize(tokenizer, df.iloc[i], x_dataFieldName, targetDimension)
        data['input_ids'].append(tuple[0])
        data['attention_mask'].append(tuple[1])
        data['token_type_ids'].append(tuple[2])
        data['label'].append(df.iloc[i]['label'])
        
    datadf = pd.DataFrame(data)
    print(datadf.info())
    print(datadf.head())
    print()
    
    # dataloader setting
    print("Torch Dataset / Torch DataLoader instantiating...")
    dataset = BERTDataset(data)
    dataLoader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle = True, 
                            # collate_fn=lambda x:x # 배치 리스트 요소를 데이터 개별 인스턴스로 세팅
                            )
    print("done!")
    print()
    return dataLoader 

def vectorize(vectorizer, data, field, length):
    
    vectorized = vectorizer(data[field])
        
    if len(vectorized['input_ids']) >= length:
        inputIds = vectorized['input_ids'][:length-1]
        inputIds.append(3)
    else:
        inputIds = [ 0 for x in range(length)]
        inputIds[: len(vectorized['input_ids'])] = vectorized['input_ids']
        
    if len(vectorized['attention_mask']) >= length:
        attentionMask = vectorized['attention_mask'][:length]
    else:
        attentionMask = [ 0 for x in range(length)]
        attentionMask[: len(vectorized['attention_mask'])] = vectorized['attention_mask']
                
    if len(vectorized['token_type_ids']) >= length:
        tokenTypeIds = vectorized['token_type_ids'][:length]
    else:
        tokenTypeIds = [ 0 for x in range(length)]
        tokenTypeIds[: len(vectorized['token_type_ids'])] = vectorized['token_type_ids']

    # print('\n', len(inputIds), len(attentionMask), len(tokenTypeIds))
    return (inputIds, attentionMask, tokenTypeIds)

class BERTDataset(Dataset):
    
    def __init__(self, dataset):

        self.dataset = dataset # python dictionary type dataset
        
    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환값:
            데이터 포인트의 특성(x_data)과 레이블(y_target) 등으로 이루어진 딕셔너리
        """
        
        input_ids = \
            torch.LongTensor(np.array(self.dataset['input_ids'][index]))

        attention_mask = \
            torch.LongTensor(np.array(self.dataset['attention_mask'][index]))
        
        token_type_ids = \
            torch.LongTensor(np.array(self.dataset['token_type_ids'][index]))
        
        label = \
            torch.LongTensor(np.array([self.dataset['label'][index]]))

        # print()
        # print("index and label: ")
        # print(index)       
        # print(label)
        # print()
        
        return [input_ids, attention_mask, token_type_ids, label]

    def __len__(self):
        return (len(self.dataset['label']))
      
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = targetDimension,
                 num_classes=num_labels,
                 dr_rate=dr_rate,
                 ):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def forward(self, input_ids, attention_mask, token_type_ids): 
        # overriding __call__() function
        
        pooler = self.bert(input_ids=input_ids, 
                           token_type_ids = token_type_ids, 
                           attention_mask = attention_mask)
        
        # print(pooler)
        # input()
        
        # if self.dr_rate:
        #     out = self.dropout(pooler)
        # else:
        #     out = pooler
            
        # return self.classifier(out)
        
        return pooler

#정확도 측정을 위한 함수 정의
def calc_accuracy(logitsTensorList,labelList):
    max_vals, max_indices = torch.max(logitsTensorList, 1) #
    train_acc = (max_indices == labelList).sum().data.cpu().numpy()/max_indices.size()[0]
    # 두 리스트의 같은 위치의 요소를 비교해서 조건식을 충족하는 경우에는 그 충족 횟수의 합계를 내고
    # 그 합계를 리스트의 요소 갯수로 나누어 점수를 구함
    return train_acc
    
def predict(predict_sentence):
        
    data = {'precSentences': predict_sentence, 'label': 0}
    dataloader= dataloader_factory(pd.DataFrame(data), x_dataFieldName, targetDimension, 1)
    
    bertmodel.eval()

    for batch_id, item in enumerate(dataloader):
        
        input_ids, attention_mask, token_type_ids = item
        
        out = bertmodel(input_ids, attention_mask, token_type_ids)

        test_eval=[]
        for i in out.logits:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("민사")
            elif np.argmax(logits) == 1:
                test_eval.append("행정")
            elif np.argmax(logits) == 2:
                test_eval.append("형사")
            elif np.argmax(logits) == 3:
                test_eval.append("특허")
            elif np.argmax(logits) == 4:
                test_eval.append("가정")
            elif np.argmax(logits) == 5:
                test_eval.append("신청")
            elif np.argmax(logits) == 6:
                test_eval.append("특별")

        print(">> 입력하신 내용은 " + test_eval[0] + " 사건에 해당합니다.")
        
if __name__ == '__main__' :
    
    # model loading
    print("model loading...")
    model = \
        BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=num_labels)
    with open('kobertbasev1model.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    with open('kobertbasev1model.pickle', 'rb') as f:
        model = pickle.load(f)
    print()
         
    # tokenizer loading
    print("tokenizer loading...")
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    with open('kobertbasev1tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)
    with open('kobertbasev1tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    print()
 
    # dataset loading (DataFrame)
    print("dataset loading....")
    dfFinal = pd.read_csv(dataFileName)
    dfFinal['label']=None
    print()
    
    # labeling
    print("labeling...")
    vocabLabel = Vocabulary()
    vocabLabel.add_many(dfFinal[y_dataFieldName].tolist())
    print('num of case sorts: ')
    print(len(vocabLabel))
    print()
    for i in range(len(vocabLabel)):
        print("Category Number: ")
        print(i)
        print("Case Sort Code: ")
        print(vocabLabel.lookup_index(i))
        print()
    # for i in range(len(df)):
    #     temp = vocabLabel.lookup_token(df.iloc[i][y_dataFieldName]) # dataframe.iloc[i] -> Series
    #     df.loc[i, 'label'] = temp # dataframe.iloc[i] -> Series with NO WARNING
    # print(df.info())
    # with open('labeleddf.pickle', 'wb') as f:
    #     pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    with open('labeleddf.pickle', 'rb') as f:
        df = pickle.load(f)
    print(df['label'].value_counts())
    print()
    print("label truncating...")
    df = df.drop(df[df['label'] == 2].index)
    df = df.drop(df[df['label'] == 1].index)
    df = df.drop(df[df['label'] == 5].index)
    df = df.drop(df[df['label'] == 9].index)
    df = df.drop(df[df['label'] == 3].index)
    df = df.drop(df[df['label'] == 6].index)
    print(df['label'].value_counts())
    print()
    print("label to index...")
    df.loc[df['label']==4,  'label']= 'civil'
    df.loc[df['label']==11, 'label']= 'admin'
    df.loc[df['label']==12, 'label']= 'crimi'
    df.loc[df['label']==10, 'label']= 'paten'
    df.loc[df['label']==0,  'label']= 'famil'
    df.loc[df['label']==8,  'label']= 'apply'
    df.loc[df['label']==7,  'label']= 'speci'
    print(df['label'].value_counts())
    print()
    df.loc[df['label']=='civil', 'label']= 0
    df.loc[df['label']=='admin', 'label']= 1
    df.loc[df['label']=='crimi', 'label']= 2
    df.loc[df['label']=='paten', 'label']= 3
    df.loc[df['label']=='famil', 'label']= 4
    df.loc[df['label']=='apply', 'label']= 5
    df.loc[df['label']=='speci', 'label']= 6
    print(df['label'].value_counts())
    print()
    
    # dataset splitting 
    print("train / test datasets splitting...")
    xTrain, xTest, yTrain, yTest = \
        train_test_split(df['precSentences'], df['label'], \
        test_size=0.2, random_state= 42, shuffle=True, stratify=df['label'])
    dfTrain = pd.concat((xTrain, yTrain), axis = 1)
    dfTest = pd.concat((xTest, yTest), axis = 1)
    print(dfTrain.info())
    print(dfTrain.head())
    print()
    print(dfTest.info())
    print(dfTest.tail())
    print()
    
    # dataset sampling
    print("train / test dataset sampling...")
    dfTrainSample = dfTrain.sample(frac=sampleRatio, random_state=999)
    dfTestSample = dfTest.sample(frac=sampleRatio, random_state=999)
    print()
    
    print("DATA VECTORIZING AND LOADING ON DATALOADER OBJECT...")
    print()
    ###########################################
    print("train dataset...")
    print(dfTrainSample.info())
    print()
    train_loader = dataloader_factory(dfTrainSample, 
                                      x_dataFieldName, 
                                      targetDimension,
                                      batch_size=batch_size
                                      )    
    ###########################################
    print("test dataset...")
    print(dfTestSample.info())
    print()
    test_loader = dataloader_factory(dfTestSample, 
                                     x_dataFieldName, 
                                     targetDimension,
                                     batch_size=batch_size
                                     )    
    ###########################################
    
    # TRAINING
    print("Press Enter Key for training your model...")
    input()
    
    # model object setting
    bertmodel = BERTClassifier(model)
    
    # optimizer와 scheduler 설정
    t_total = len(train_loader) * epochs
    warmup_steps = int(t_total * warmup_ratio)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bertmodel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in bertmodel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = \
        get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_total)

    # loss function setting
    loss_fn = nn.CrossEntropyLoss()
   
    # GO! 
    train_history=[]
    test_history=[]
    loss_history=[]
    
    for e in range(epochs):
        
        train_acc = 0.0
        test_acc = 0.0
        
        #TRAINING
        bertmodel.train()
        for batch_id, batch in enumerate(tqdm(train_loader)):
            
            if batch_id % log_interval == 0 : 
                print(f"Epoch : {e+1} in {epochs} / Minibatch Step : {batch_id}")

            # print(type(item))
            # print(item)
            # print(item.__dir__)
                        
            input_ids, attention_mask, token_type_ids, label = batch
            label = label.squeeze(1)
            
            # print(input_ids)
            # print(attention_mask)
            # print(token_type_ids)
            # print(label)
            
            # input()
            
            #1 gradient를 0으로 초기화
            optimizer.zero_grad()
            
            #2 출력 계산
            out = bertmodel(input_ids, 
                            attention_mask = attention_mask, 
                            token_type_ids = token_type_ids, 
                            # labels=label
                            ) 
            # print()
            # print("out:")
            # print(out)
            # print(type(out))
            # print("label:")
            # print(label)
            # print(type(label))
                        
            #3 손실 계산
            loss = loss_fn(out.logits, label)
            # loss = out.loss
            # print(loss)
            # input()
            
            #4 손실로 gradient 계산
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(bertmodel.parameters(), max_grad_norm)
            
            #5 계산된 gradient로 가중치를 갱신
            optimizer.step()
            
            #6 Update learning rate schedule
            scheduler.step()  
            
            #7 정확도 계산
            train_acc += calc_accuracy(out.logits, label)
            # train_acc += loss.item()
                        
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                train_history.append(train_acc / (batch_id+1))
                loss_history.append(loss.data.cpu().numpy())
                
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
        # EVALUATING
        bertmodel.eval()
        for batch_id, item in enumerate(tqdm(test_loader)):
            
            if batch_id % log_interval == 0 : 
                print(f"Epoch : {e+1} in {epochs} / Minibatch Step : {batch_id}")
                        
            input_ids, attention_mask, token_type_ids, label = batch
            label = label.squeeze(1)
        
            out = bertmodel(input_ids, attention_mask, token_type_ids)
            
            test_acc += calc_accuracy(out.logits, label)

        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        test_history.append(test_acc / (batch_id+1))
    
    #질문 무한반복하기! 0 입력시 종료
    while True :
        sentence = input("분류를 위한 사건 텍스트를 입력한 후 엔터키를 누르십시오: ")
        if sentence == "0" :
            break
        predict(sentence)
        print("\n")