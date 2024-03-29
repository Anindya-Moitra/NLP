# An unstructured text classifier built from the scratch. The text can be categorized into
# two classes, so it's a binary classification task.

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
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterGrid


# Load and preprocess the data.

# Deal with class-imbalance.
data_imbalanced = pd.read_csv('unstruct_text.csv')
data_class_1 = data_imbalanced.loc[data_imbalanced['Class'] == 1]
data_class_0 = data_imbalanced.loc[data_imbalanced['Class'] == 0]
data_class_0_sample = data_class_0.sample(n=254, random_state=10)

data = data_class_1.append(data_class_0_sample, ignore_index=True)
data = data.sample(frac=1, random_state=20).reset_index(drop=True)  # Ramdomly shuffle the rows of the previous df

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


data['Cleaned'] = data['Note'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z0-9]", " ", x).split() if i not in stop_words_custom]).lower())

X, y = data['Cleaned'], data['Target']

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)  # Initiate cross-validation object

pipeline_svc = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words=stop_words_custom, sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=25000)),
                         ('clf', LinearSVC(C=4.0, penalty='l1', max_iter=1000, dual=False, random_state=0))
                         ])

scores_svc = cross_val_score(pipeline_svc, X, y, cv=cv)

print("Linear SVC Mean Accuracy with 95%% CI: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))

grid = ParameterGrid({"max_samples": [0.6, 0.8], "n_estimators": [11, 21, 31, 41, 51, 61, 71, 81, 91, 101],
                      "bootstrap_features": [True, False]})


# Find an optimal set of parameters for the classification model using a parameter grid.
for param in grid:
    print('\n')
    print(param)
    pipeline_rf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 4), stop_words=stop_words_custom, sublinear_tf=True)),
                            ('chi',  SelectKBest(chi2, k=25000)),
                            ('clf', BaggingClassifier(random_state=20, **param))  # By default, the base estimator is a decision tree
                            ])
    scores_rf = cross_val_score(pipeline_rf, X, y, cv=cv)
    print("Random Forest Mean Accuracy with 95%% CI: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))
    print('\n')
    print("=================================================================================")
