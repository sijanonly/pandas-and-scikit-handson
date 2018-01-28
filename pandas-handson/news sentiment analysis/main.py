# -*- coding: utf-8 -*-
# @Author: leapfrong
# @Date:   2017-11-03 12:26:44
# @Last Modified by:   leapfrong
# @Last Modified time: 2017-11-14 13:55:15
import csv
import pandas as pd
import pickle


def main():
    with open('linear.bin', 'rb') as model_file, open('news_file_sample.csv', 'r') as data_file:
        data_struct = pickle.load(model_file)
        vectorizer, model = data_struct['vectorizer'], data_struct['model']

        reader = csv.DictReader(data_file, delimiter='|')
        data = [row['article_name'] for row in reader]

        # vectorize the raw data
        new_data_vectors = vectorizer.transform(data)
        predictions = model.predict(new_data_vectors)

    values = pd.Series(['positive' if prediction ==
                        '1' else 'negative' for prediction in predictions])
    csv_input = pd.read_csv('news_file_sample.csv', delimiter="|")
    csv_input['sentiment'] = values
    csv_input.to_csv('all_sentiment.csv', index=False)


if __name__ == '__main__':
    main()
