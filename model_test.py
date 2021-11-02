import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt
import torch
import json
import pickle

with open("./vectorizer.json","r") as fp:
    vectorizer=json.load(fp)

with open("./model.pth","r") as model:
    classifier=model

with open('word2vec.model', 'rb') as f:
    wvmodel = pickle.load(f)



def ko_sentences_preproc(lines) :

    lines=lines.split("lnfd")
    # print(f'totally {len(lines)} lines')
    i = 0
    sentences = []
    okt = Okt()
    for line in lines:
        if not line: 
            print(f'no item at {i}')
            continue
        if i % 500 == 0:
            print("current(every 500) - " + str(i+1))
        i += 1

        nonhangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
        line = nonhangul.sub(' ', line) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

        # 형태소 분석
        malist = okt.pos(line, norm=True, stem=True)
        # 필요한 어구만 대상으로 하기
        r = []
        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외 
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                r.append(word[0])
        sentences.append(" ".join(r))
    # print(f'totally {i} lines')

    return " ".join(sentences)

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

text= input()
print("법률문서를 입력하세요: ", text)
test_review =  ko_sentences_preproc(text)

prediction = predict_rating(test_review, classifier, vectorizer)
print("{} -> {}(p={:0.2f})".format(test_review, prediction['rating'], prediction['probability']))