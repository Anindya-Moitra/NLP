# Multilabel text classification
# This program builds a multilabel text classifier from the scratch. Each document in this task has two target classes.

import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterGrid


data = pd.read_csv('documents.csv')

# stemmer = PorterStemmer()
stemmer = SnowballStemmer('english')
# lemmatizer = WordNetLemmatizer()
stop_words_custom = set(stopwords.words('english'))

stop_words_custom.remove('no')
stop_words_custom.remove('not')
stop_words_custom.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                   'ok', 'w', 'md', 'p', 'iii', 'iv', 'v', 'vi', 'vii', 'aa', 'aaa', 'aair', 'aaox', 'aap',
                   'aaron', 'ab', 'bs', 'igg', 'abc', 'abd', 'abdo', 'abfp', 'abg', 'abi', 'abid', 'abil',
                   'abilen', 'abilifi', 'abl'])

print(stop_words_custom)

data['cleaned'] = data['note'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words_custom]).lower())

X, y = data['cleaned'], data[['class_0', 'class_1']]
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

pipeline1 = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words=stop_words_custom, sublinear_tf=True)),
                      ('chi',  SelectKBest(chi2, k=25000)),
                      ('clf', OneVsRestClassifier(LinearSVC(C=4.0, penalty='l1', max_iter=1000, dual=False, random_state=0)))])

scores = cross_val_score(pipeline1, X, y, cv=cv)

print("Linear SVC Mean Accuracy with 95%% CI: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

grid = ParameterGrid({"max_samples": [0.6, 0.7, 0.8, 0.9, 1.0],
                      "n_estimators": [101, 201, 301, 401, 501, 601, 701, 801, 901, 1001],
                      "bootstrap_features": [True, False]})


# Hyper-parameter tuning for the classification model using a parameter grid.
for param in grid:
    print(param)
    pipeline2 = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words=stop_words_custom, sublinear_tf=True)),
                          ('chi',  SelectKBest(chi2, k=25000)),
                          ('clf', OneVsRestClassifier(BaggingClassifier(random_state=20, **param)))  # By default, the base estimator is a decision tree
                         ])

    scores = cross_val_score(pipeline2, X, y, cv=cv)
    print("Mean Accuracy with 95%% CI: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("=================================================================================")
