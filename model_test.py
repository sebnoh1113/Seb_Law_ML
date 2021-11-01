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

test_review = 
prediction = predict_rating(test_review, classifier, vectorizer)
print("{} -> {}(p={:0.2f})".format(test_review, prediction['rating'], prediction['probability']))