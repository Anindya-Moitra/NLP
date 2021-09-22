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
