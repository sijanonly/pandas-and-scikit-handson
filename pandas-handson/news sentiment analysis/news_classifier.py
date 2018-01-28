
import pickle
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

TRAIN_CORPUS_FILES = ['amazon.txt', 'imdb.txt']
TEST_CORPUS_FILES = ['yelp.txt']


class NewsClassification:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        self.train_vectors = None
        self.test_vectors = None

        # self.vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
        self.vectorizer = TfidfVectorizer()
        self.model = None

    def read_news_corpus(self, file):
        with open('news_corpus/%s' % (file), 'r') as f:
            return f.readlines()

    def prepare_data(self, prepared_for='training'):
        data = []
        labels = []
        if prepared_for.strip().lower() == 'training':
            CORPUS_FILES = TRAIN_CORPUS_FILES
        else:
            CORPUS_FILES = TEST_CORPUS_FILES

        for train_file in CORPUS_FILES:
            rows = self.read_news_corpus(train_file)
            for row in rows:
                line = row.strip().rsplit('\t', 1)
                try:
                    text = line[0]
                    label = line[1].replace('\t', '').strip()
                    data.append(text)
                    labels.append(label)
                except Exception:
                    pass
        return data, labels

    def prepare_feature_vectors(self):
        # Train the feature vectors
        self.train_vectors = self.vectorizer.fit_transform(self.train_data)
        # Apply model on test data
        # : since they have already been fit to the training set
        self.test_vectors = self.vectorizer.transform(self.test_data)

    def prepare_model(self, classifier_type='svm', kernel='linear'):
        self.model = svm.SVC(kernel=kernel)
        self.model.fit(self.train_vectors, self.train_labels)

        # let's save our model using pickle.
        # we will using this model later for unseen data as well.
        data_struct = {'vectorizer': self.vectorizer, 'model': self.model}
        with open('%s.bin' % kernel, 'wb') as f:
            pickle.dump(data_struct, f)

        # prediction = self.model.predict(self.test_vectors)
        # return prediction


if __name__ == '__main__':
    news_classifier = NewsClassification()
    news_classifier.train_data, news_classifier.train_labels =\
        news_classifier.prepare_data()
    news_classifier.test_data, news_classifier.test_labels =\
        news_classifier.prepare_data('testing')

    news_classifier.prepare_feature_vectors()
    news_classifier.prepare_model(kernel='linear')
    prediction = news_classifier.model.predict(news_classifier.test_vectors)
    with open('news_corpus/yelp.txt', 'r') as f:
        test_data = f.readlines()
        test_data = [row.replace('\n', '') for row in test_data]
        reader = csv.reader(test_data, delimiter='\t')
        final_data = []
        for row, predicted_value in zip(reader, prediction):
            row.append(predicted_value)
            final_data.append(row)
    df = pd.DataFrame.from_records(
        final_data,
        columns=['title', 'actual_prediction', 'predicted_value'])
    print(df.to_string())
    # print('test data', test_data)
    #     print('test_data            actual_value          predicted_value')
    #     for line, predict_value in zip(test_data, prediction):
    #         print(line, predict_value)
    # predicted_test_data = pd.read_csv('news_corpus/yelp.txt', delimiter=",")
    # print(predicted_test_data.to_string())
    # predicted_test_data['sentiment'] = prediction
    # print(predicted_test_data.to_string())
    # print (classification_report(news_classifier.test_labels, prediction))
    # docs_new = ['this is new title']
    # new_data_vectors = news_classifier.vectorizer.transform(docs_new)
    # print('new predictin', news_classifier.model.predict(new_data_vectors))
    # print(new_data_vectors)

    # predicted = clf.predict(X_new_tfidf)
