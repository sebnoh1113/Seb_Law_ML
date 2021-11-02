import pandas as pd
import numpy as np
import re
from konlpy.tag import Okt


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





df=pd.read_pickle('e:/20211026/df_corpus_full.pickle')

df_for_konlpy=df.loc[ df['reasoning']!=0 , ['case_sort','reasoning'] ] #corpus_full은 nan값 처리 완료 된 상태

df_for_konlpy['precSentences'] = df_for_konlpy['reasoning'].apply(ko_sentences_preproc)

df_for_konlpy.to_csv("e:/dfFinal.csv", encoding='utf-8-sig')


print("종료")