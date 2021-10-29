from argparse import Namespace
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from konlpy.tag import Okt
from gensim.models import word2vec

import pickle
import collections
import json
import os
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

args = Namespace(
    
    seed=1337,
    
    raw_train_dataset_csv="./dfFinal.csv",
    proportion_subset_of_train=1.0,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="./dfFinal_splits.csv",
    
    frequency_cutoff=25,
    # frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='./dfFinal_splits.csv',
    save_dir='./',
    vectorizer_file='vectorizer.json',

    # 모델 하이퍼파라미터
    # hidden_dim=100,
    num_channels=256,
    # 훈련 하이퍼파라미터
    learning_rate=0.001,
    batch_size=16,
    # batch_size=128,
    num_epochs=100,
    early_stopping_criteria=5,
    dropout_p=0.1,

    # 실행 옵션
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)

print("# word2vec retrieving")
with open('word2vec.model', 'rb') as f:
    wvmodel = pickle.load(f)
# def ko_sentences_preproc(lines: list) -> list : # from konlpy.tag import Okt
#         print(f'totally {len(lines)} lines')
#         i = 0
#         sentences = []
#         okt = Okt()
#         for line in lines:
#             if not line: 
#                 print(f'no item at {i}')
#                 continue
#             if i % 500 == 0:
#                 print("current(every 500) - " + str(i+1))
#             i += 1

#             nonhangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
#             line = nonhangul.sub(' ', line) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

#             # 형태소 분석
#             malist = okt.pos(line, norm=True, stem=True)
#             # 필요한 어구만 대상으로 하기
#             r = []
#             for word in malist:
#                 # 어미/조사/구두점 등은 대상에서 제외 
#                 if not word[1] in ["Josa", "Eomi", "Punctuation"]:
#                     r.append(word[0])
#             sentences.append(" ".join(r))
#         print(f'totally {i} lines')
#         return sentences

def final_reviews_maker(args):
    # 원본 데이터를 읽습니다
    train_reviews = pd.read_csv(args.raw_train_dataset_csv) # 한글 전처리된 csv
    train_reviews = train_reviews.dropna()
    header_dict={'case_sort':'rating', 'precSentences':'review'}
    train_reviews.rename(columns=header_dict,inplace=True)
    print("\n train reviews based on dfFinal csv: \n")
    print(train_reviews.info())
  
    # 클래스 비율이 바뀌지 않도록 서브셋을 만듭니다
    by_rating = collections.defaultdict(list)
    for _, row in train_reviews.iterrows():
        by_rating[row.rating].append(row.to_dict())
    review_subset = []
    for _, item_list in sorted(by_rating.items()):
        n_total = len(item_list)
        n_subset = int(args.proportion_subset_of_train * n_total)
        review_subset.extend(item_list[:n_subset])
    review_subset = pd.DataFrame(review_subset)
    print("\n review subset based on train reviews: \n")
    print(train_reviews.info())
    print(review_subset.head())

    # 훈련, 검증, 테스트를 만들기 위해 클래스를 기준으로 나눕니다
    by_rating = collections.defaultdict(list)
    for _, row in review_subset.iterrows():
        by_rating[row.rating].append(row.to_dict())
    final_list = []
    np.random.seed(args.seed)
    for _, item_list in sorted(by_rating.items()):
        np.random.shuffle(item_list)
        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)
        n_test = int(args.test_proportion * n_total)
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
        final_list.extend(item_list)

    # 분할 데이터를 데이터 프레임으로 만듭니다
    final_reviews = pd.DataFrame(final_list)
    print("\n final review based on review subset: \n")
    print(final_reviews.info())
    print(final_reviews.head())
   
    final_reviews.to_csv(args.output_munged_csv, index=False) # 전처리와 분할된 데이터프레임
    return final_reviews

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        print("neither folder nor file, therefore creating...")
        os.makedirs(dirpath)
        
class Vocabulary(object):
    """ 매핑을 위해 텍스트를 처리하고 어휘 사전을 만드는 클래스 """

    @classmethod
    def from_serializable(cls, contents):
        """ 직렬화된 딕셔너리에서 Vocabulary 객체를 만듭니다 """
        return cls(**contents)

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
            add_unk (bool): UNK 토큰을 추가할지 지정하는 플래그
            unk_token (str): Vocabulary에 추가할 UNK 토큰
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 
        
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
        """ 토큰 리스트를 Vocabulary에 추가합니다.
        
        매개변수:
            tokens (list): 문자열 토큰 리스트
        반환값:
            indices (list): 토큰 리스트에 상응되는 인덱스 리스트
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """ 토큰에 대응하는 인덱스를 추출합니다.
        토큰이 없으면 UNK 인덱스를 반환합니다.
        
        매개변수:
            token (str): 찾을 토큰 
        반환값:
            index (int): 토큰에 해당하는 인덱스
        노트:
            UNK 토큰을 사용하려면 (Vocabulary에 추가하기 위해)
            `unk_index`가 0보다 커야 합니다.
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
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
            raise KeyError("Vocabulary에 인덱스(%d)가 없습니다." % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

    def to_serializable(self):
        """ 직렬화할 수 있는 딕셔너리를 반환합니다 """
        return {'token_to_idx': self._token_to_idx, 
                'add_unk': self._add_unk, 
                'unk_token': self._unk_token}

class ReviewVectorizer(object):
    """ 어휘 사전을 생성하고 관리합니다 """
    
    @classmethod ###################################################################
    def from_dataframe(cls, review_df, cutoff=25):
        """ 데이터셋 데이터프레임에서 Vectorizer 객체를 만듭니다
        
        매개변수:
            review_df (pandas.DataFrame): 리뷰 데이터셋
            cutoff (int): 빈도 기반 필터링 설정값
        반환값:
            ReviewVectorizer 객체
        """
        # 초기화 with empty dict and 0
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)
        max_review_length=0

        # 점수(사건 분류)를 추가합니다
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # count > cutoff인 단어를 추가합니다
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)
        for _, row in review_df.iterrows():
            max_review_length = max(max_review_length, len(row.review.split(" ")))
        return cls(review_vocab, rating_vocab, max_review_length) #####################################
    
    @classmethod
    def from_serializable(cls, contents):
        """ 직렬화된 딕셔너리에서 ReviewVectorizer 객체를 만듭니다
        
        매개변수:
            contents (dict): 직렬화된 딕셔너리
        반환값:
            ReviewVectorizer 클래스 객체
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab =  Vocabulary.from_serializable(contents['rating_vocab'])
        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab, max_review_length=contents['max_review_length'])

    def __init__(self, review_vocab, rating_vocab, max_review_length):
        """
        매개변수:
            review_vocab (Vocabulary): 단어를 정수에 매핑하는 Vocabulary
            rating_vocab (Vocabulary): 클래스 레이블을 정수에 매핑하는 Vocabulary
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
        self._max_review_length = max_review_length

    def vectorize(self, review):
        """ 리뷰에 대한 웟-핫 벡터를 만듭니다
        매개변수:
            review (str): 리뷰
        반환값:
            one_hot (np.ndarray): 원-핫 벡터
        """
        matrix_size=  (len(self.review_vocab), self._max_review_length) # 한 열이 한 단어 열의 개수는 문장 길이
        one_hot_matrix = np.zeros(matrix_size, dtype=np.float32)
        
        # return one_hot_matrix
        for position_index, token in enumerate(review.split(" ")):
            token_index = self.review_vocab.lookup_token(token)
            one_hot_matrix[token_index][position_index] = 1

        return one_hot_matrix

    def wvectorize(self, review, wvmodel):
        """ 리뷰에 대한 웟-핫 벡터를 만듭니다
        매개변수:
            review (str): 리뷰
        반환값:
            one_hot (np.ndarray): 원-핫 벡터
        """
        matrix_size=  (wvmodel.vector_size, self._max_review_length) # 한 열이 한 단어 열의 개수는 문장 길이
        one_hot_matrix = np.zeros(matrix_size, dtype=np.float32)
        
        # return one_hot_matrix
        for i, token in enumerate(review.split(" ")):
            if i < matrix_size:
                try:
                    wvector = wvmodel.wv[token]
                except:
                    wvector = np.zeros(100, dtype=np.float32)
                one_hot_matrix[:, i]= wvector

        return one_hot_matrix
    
    def to_serializable(self):
        """ 캐싱을 위해 직렬화된 딕셔너리를 만듭니다
        
        반환값:
            contents (dict): 직렬화된 딕셔너리
        """
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable(),
                'max_review_length': self._max_review_length}


class ReviewDataset(Dataset):
    
    @classmethod ####################################################################
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """ 데이터셋을 로드하고 새로운 ReviewVectorizer 객체를 만듭니다
        
        매개변수:
            review_csv (str): 데이터셋의 위치
        반환값:
            ReviewDataset의 인스턴스
        """
        review_df = pd.read_csv(review_csv)
        train_review_df = review_df[review_df.split=='train']
        return cls(review_df, ReviewVectorizer.from_dataframe(train_review_df)) ##########
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):
        """ 데이터셋을 로드하고 새로운 ReviewVectorizer 객체를 만듭니다.
        캐시된 ReviewVectorizer 객체를 재사용할 때 사용합니다.
        
        매개변수:
            review_csv (str): 데이터셋의 위치
            vectorizer_filepath (str): ReviewVectorizer 객체의 저장 위치
        반환값:
            ReviewDataset의 인스턴스
        """
        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)
    
    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """ 파일에서 ReviewVectorizer 객체를 로드하는 정적 메서드
        
        매개변수:
            vectorizer_filepath (str): 직렬화된 ReviewVectorizer 객체의 위치
        반환값:
            ReviewVectorizer의 인스턴스
        """
        with open(vectorizer_filepath) as fp:
            return ReviewVectorizer.from_serializable(json.load(fp))

    def __init__(self, review_df, vectorizer):
        """
        매개변수:
            review_df (pandas.DataFrame): 데이터셋
            vectorizer (ReviewVectorizer): ReviewVectorizer 객체
        """
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train') # "train", "val", "test" 중 하나

        # 클래스 별 데이터 포인트를 균등하게 맞추어주는 가중치 계산 => 크로스엔트로피 함수의 파라미터로 활용
        class_counts = review_df.rating.value_counts().to_dict() # Series - key : rating / value : no of reviews with rating 
        def sort_key(item):
            return self._vectorizer.rating_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key) # items()는 튜플의 리스트를 반환
        frequencies = [count for _, count in sorted_counts] # 가나다 순 정렬된 후임
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32) # 해당 분류에 속한 데이터 포인트 개수가 많으면 이에 역비례하여 작은 수가 가중치로 설정됨

    def set_split(self, split="train"):
        """ 데이터프레임에 있는 열을 사용해 분할 세트를 선택합니다 
        
        매개변수:
            split (str): "train", "val", "test" 중 하나
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def save_vectorizer(self, vectorizer_filepath):
        """ ReviewVectorizer 객체를 json 형태로 디스크에 저장합니다
        
        매개변수:
            vectorizer_filepath (str): ReviewVectorizer 객체의 저장 위치
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ 벡터 변환 객체를 반환합니다 """
        return self._vectorizer

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ 파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환값:
            데이터 포인트의 특성(x_data)과 레이블(y_target)로 이루어진 딕셔너리
        """
        row = self._target_df.iloc[index]

        review_matrix = \
            self._vectorizer.wvectorize(row.review, wvmodel) # vector 아니라 matrix 반환 (review에 해당)

        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.rating) # vector 아니라 정수 반환

        return {'x_data': review_matrix,
                'y_target': rating_index}

    def get_num_batches(self, batch_size):
        """ 배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다
        
        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size

class ReviewClassifier(nn.Module):

    def __init__(self, initial_num_channels, num_classes, num_channels):
    # def __init__(self, initial_num_channels, num_classes, num_channels):
        """
        매개변수:
            initial_num_channels (int): 입력 특성 벡터의 크기
            num_classes (int): 출력 예측 벡터의 크기
            num_channels (int): 신경망 내부에 사용될 채널(뉴런) 크기
            ########################
            classifier = ReviewClassifier(initial_num_channels=len(vectorizer.review_vocab), num_classes=len(vectorizer.rating_vocab), num_channels=args.num_channels) ########################
        """
        super(ReviewClassifier, self).__init__()
       
        self.convnet1 = nn.Sequential( # out channels / kernel size / stride 등은 서로 독립적으로 정해질 수 있음 / 목표는 열 개수가 1이 되도록 하는 것으로 이는 stride가 주로 영향을 줌 # shape -> 12622, 26408
            nn.Conv1d(in_channels=initial_num_channels, 
                      out_channels=80, kernel_size=15, stride = 7),
              )
        self.convnet2 = nn.Sequential( # out channels / kernel size / stride 등은 서로 독립적으로 정해질 수 있음 / 목표는 열 개수가 1이 되도록 하는 것으로 이는 stride가 주로 영향을 줌 # shape -> 12622, 26408
            nn.Conv1d(in_channels=80, out_channels=60, 
                      kernel_size=10, stride=7),
            nn.ELU(),
            nn.Conv1d(in_channels=60, out_channels=40, 
                      kernel_size=7, stride=5),
            nn.ELU(),
            nn.Conv1d(in_channels=40, out_channels=20, 
                      kernel_size=7, stride=5),          
            nn.ELU(),
            nn.Conv1d(in_channels=20, out_channels=15, 
                      kernel_size=7, stride=5),
            nn.ELU(),
            nn.Conv1d(in_channels=15, out_channels=8, 
                      kernel_size=7, stride=5),
            nn.ELU(),
            nn.Conv1d(in_channels=8, out_channels=256, 
                      kernel_size=4, stride=1),                   
             nn.ELU(),
              )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x_data, apply_softmax=False):
        """모델의 정방향 계산
        
        매개변수:
            x_surname (torch.Tensor): 입력 데이터 텐서. 
                x_surname.shape은 (batch, initial_num_channels, max_surname_length)입니다.
            apply_softmax (bool): 소프트맥스 활성화 함수를 위한 플래그
                크로스-엔트로피 손실을 사용하려면 False로 지정해야 합니다.
        반환값:
            결과 텐서. tensor.shape은 (batch, num_classes)입니다.
        """
        print("\n \nshapes---------- \n")
        print(x_data.shape)
        temp = self.convnet1(x_data)
        print(temp.shape)
        features = self.convnet2(temp)
        print(features.shape)
        features = features.squeeze(dim=2)
        print(features.shape)
        print("shapes---------- \n")
        # torch.Size([10, 4667, 15604])
        # torch.Size([10, 256, 1])
        # torch.Size([10, 256])
        # torch.Size([2, 12622, 26408])
        # torch.Size([2, 256, 2])
        # torch.Size([2, 256, 2])
        
        prediction_vector = self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    파이토치 DataLoader를 감싸고 있는 제너레이터 함수.
    걱 텐서를 지정된 장치로 이동합니다.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, _ in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """ 리뷰 점수 예측하기
    
    매개변수:
        review (str): 리뷰 텍스트
        classifier (ReviewClassifier): 훈련된 모델
        vectorizer (ReviewVectorizer): Vectorizer 객체
        decision_threshold (float): 클래스를 나눌 결정 경계
    """
    #review = ko_sentences_preproc(review)
    
    vectorized_review = torch.tensor(vectorizer.wvectorize(review, wvmodel))
    result = classifier(vectorized_review.view(1, -1), apply_softmax=True)
    
    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_rating = vectorizer.rating_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {'rating': predicted_rating, 'probability': probability_value}
 
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """ 훈련 상태를 업데이트합니다.

    Components:
     - 조기 종료: 과대 적합 방지
     - 모델 체크포인트: 더 나은 모델을 저장합니다

    :param args: 메인 매개변수
    :param model: 훈련할 모델
    :param train_state: 훈련 상태를 담은 딕셔너리
    :returns:
        새로운 훈련 상태
    """

    # 적어도 한 번 모델을 저장합니다
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # 손실이 나빠지면
        if loss_t >= train_state['early_stopping_best_val']:
            # 조기 종료 단계 업데이트
            train_state['early_stopping_step'] += 1
        # 손실이 감소하면
        else:
            # 최상의 모델 저장
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # 조기 종료 단계 재설정
            train_state['early_stopping_step'] = 0

        # 조기 종료 여부 확인
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    y_pred_indices = y_pred.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


if __name__ == '__main__':

    final_reviews_maker(args) # 분할된 데이터프레임을 통해 csv 생성 및 저장
    
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir,
                                            args.model_state_file)
        
        print("파일 경로: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        
    # CUDA 체크
    if not torch.cuda.is_available():
        args.cuda = False
    print("CUDA 사용여부: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # 재현성을 위해 시드 설정
    set_seed_everywhere(args.seed, args.cuda)

    # 디렉토리 처리
    handle_dirs(args.save_dir)

    # 데이터셋 로드
    if args.reload_from_files:
        # 체크포인트에서 훈련을 다시 시작
        print("데이터셋과 Vectorizer를 로드합니다")
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                args.vectorizer_file) #######################
    else:
        print("데이터셋을 로드하고 Vectorizer를 만듭니다")
        # 데이터셋과 Vectorizer 만들기
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
        dataset.save_vectorizer(args.vectorizer_file)    

    # 벡토라이저/클래시파이어 로드
    vectorizer = dataset.get_vectorizer()

    # 클래시파이어 로드
    classifier = ReviewClassifier(initial_num_channels=100, num_classes=len(vectorizer.rating_vocab), num_channels=args.num_channels) 
    # classifier = ReviewClassifier(initial_num_channels=len(vectorizer.review_vocab), num_classes=len(vectorizer.rating_vocab), num_channels=args.num_channels) 
    print("\n 데이터 포인트 매트릭스의 차원: \n")
    print(f" row : {len(vectorizer.review_vocab)} \n")
    print(f" col : {vectorizer._max_review_length} \n")
    print(f" class : {len(vectorizer.rating_vocab)} \n")
    ########################
    classifier = classifier.to(args.device)
    dataset.class_weights=dataset.class_weights.to(args.device)

    # 손실함수/최적화 옵션 설정
    loss_func = nn.CrossEntropyLoss(weight=dataset.class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                    mode='min', factor=0.5,
                                                    patience=1)
    # 진행상황 tqdm 세팅
    train_state = make_train_state(args)
    epoch_bar = tqdm.std.tqdm(desc='training routine', 
                            total=args.num_epochs,
                            position=0)
    dataset.set_split('train')
    train_bar = tqdm.std.tqdm(desc='split=train',
                            total=dataset.get_num_batches(args.batch_size), 
                            position=1, 
                            leave=True)
    dataset.set_split('val')
    val_bar = tqdm.std.tqdm(desc='split=val',
                          total=dataset.get_num_batches(args.batch_size), 
                          position=1, 
                          leave=True)

    # training / validation
    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # 훈련 세트에 대한 순회

            # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            dataset.set_split('train')
            batch_generator = generate_batches(dataset, 
                                            batch_size=args.batch_size, 
                                            device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator): ########## learning #############
                # 훈련 과정은 5단계로 이루어집니다

                # --------------------------------------
                # 단계 1. 그레이디언트를 0으로 초기화합니다
                optimizer.zero_grad()

                # 단계 2. 출력을 계산합니다
                y_pred = classifier(batch_dict['x_data'].float())

                # 단계 3. 손실을 계산합니다
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 단계 4. 손실을 사용해 그레이디언트를 계산합니다
                loss.backward()

                # 단계 5. 옵티마이저로 가중치를 업데이트합니다
                optimizer.step()
                # -----------------------------------------
                
                # 정확도를 계산합니다
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # 진행 바 업데이트
                train_bar.set_postfix(loss=running_loss, 
                                    acc=running_acc, 
                                    epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # 검증 세트에 대한 순회

            # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
            dataset.set_split('val')
            batch_generator = generate_batches(dataset, 
                                            batch_size=args.batch_size, 
                                            device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # 단계 1. 출력을 계산합니다
                y_pred = classifier(batch_dict['x_data'].float())

                # 단계 2. 손실을 계산합니다
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 단계 3. 정확도를 계산합니다
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                
                val_bar.set_postfix(loss=running_loss, 
                                    acc=running_acc, 
                                    epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=classifier,
                                            train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")


    # testing
    # 가장 좋은 모델을 사용해 테스트 세트의 손실과 정확도를 계산합니다
    classifier.load_state_dict(torch.load(train_state['model_filename']))
    classifier = classifier.to(args.device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset, 
                                    batch_size=args.batch_size, 
                                    device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # 출력을 계산합니다
        y_pred = classifier(batch_dict['x_data'].float()) 

        # 손실을 계산합니다
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # 정확도를 계산합니다
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print()
    print()
    print("테스트 손실: {:.3f}".format(train_state['test_loss']))
    print()
    print("테스트 정확도: {:.2f}".format(train_state['test_acc']))

    # test_review=""#######################################################################################################
    # prediction = predict_rating(test_review, classifier, vectorizer)
    # print("{} -> {}(p={:0.2f})".format(test_review, prediction['rating'], prediction['probability']))


# PS C:\Users\imhno\Seb_Python_Projects\PyProjects\Pytorch Exercise> conda activate pytorch_cuda
#  "c:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/nlp_pipeline.py"
# 파일 경로:
#         C:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/vectorizer.json
#         C:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/model.pth
# CUDA 사용여부: True
# 데이터셋을 로드하고 Vectorizer를 만듭니다
# training routine:   0%|                                                                          | 0/100 [00:00<?, ?it/s]Traceback (most recent call last):                                                                  | 0/6 [00:00<?, ?it/s] 
#   File "c:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/nlp_pipeline.py", line 695, in <module>
#     for batch_index, batch_dict in enumerate(batch_generator): ########## learning #############
#   File "c:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/nlp_pipeline.py", line 424, in generate_batches    
#     for data_dict in dataloader:
#   File "C:\Users\imhno\anaconda3\envs\pytorch_cuda\lib\site-packages\torch\utils\data\dataloader.py", line 521, in __next__
#     data = self._next_data()
#   File "C:\Users\imhno\anaconda3\envs\pytorch_cuda\lib\site-packages\torch\utils\data\dataloader.py", line 561, in _next_data
#     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
#   File "C:\Users\imhno\anaconda3\envs\pytorch_cuda\lib\site-packages\torch\utils\data\_utils\fetch.py", line 44, in fetch 
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "C:\Users\imhno\anaconda3\envs\pytorch_cuda\lib\site-packages\torch\utils\data\_utils\fetch.py", line 44, in <listcomp>
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "c:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/nlp_pipeline.py", line 351, in __getitem__
#     self._vectorizer.wvectorize(row.review)
#   File "c:/Users/imhno/Seb_Python_Projects/PyProjects/Pytorch Exercise/nlp_pipeline.py", line 176, in vectorize
#     one_hot = np.zeros(matrix_size, dtype=np.float32)
# numpy.core._exceptions.MemoryError: Unable to allocate 329. MiB for an array with shape (5526, 15604) and data type float32
# training routine:   0%|                                                                          | 0/100 [00:01<?, ?it/s] 
# split=train:   0%|                                                                                | 0/31 [00:01<?, ?it/s] 
# split=val:   0%|                                                                                   | 0/6 [00:01<?, ?it/s] 
# PS C:\Users\imhno\Seb_Python_Projects\PyProjects\Pytorch Exercise>

# 예외가 발생했습니다. RuntimeError
# [enforce fail at ..\c10\core\CPUAllocator.cpp:79] data. DefaultCPUAllocator: not enough memory: you tried to allocate 44148584448 bytes.
#   File "C:\Users\hcjeo\VSCodeProjects\_python_practice\nlp_pipeline_case_sort.py", line 425, in generate_batches
#     for data_dict in dataloader:
#   File "C:\Users\hcjeo\VSCodeProjects\_python_practice\nlp_pipeline_case_sort.py", line 696, in <module>
#     for batch_index, batch_dict in enumerate(batch_generator): ########## learning #############