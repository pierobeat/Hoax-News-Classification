# =========== Import Packages ===========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

from jcopml.tuning.space import Real, Integer
from jcopml.utils import save_model, load_model

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

from xgboost import XGBClassifier

# =========== Variabel for stopwords + punctuation's filtering ===========
sw_indo = stopwords.words('indonesian') + list(punctuation)

# =========== Import Dataset ===========
pd.set_option("display.max_colwidth", 150)
df = pd.read_csv('./data/500_berita_indonesia.csv', delimiter=";")
df.head(10)

# =========== Function for text preprocessing ===========
def preprocessing(txt):
    nopunc = [token.lower() for token in txt if token not in punctuation]
    nopunc = ''.join(nopunc)

    tokens = word_tokenize(nopunc)

    clean_text = [token for token in tokens if token not in stopwords.words('indonesian')]

    return " ".join(clean_text)

df['berita'] = df['berita'].apply(preprocessing)

# =========== Data Splitting ===========
X = df.berita
y = df.kategori

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =========== Use Pipeline ===========
pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, stop_words=sw_indo)),
    ('algo', XGBClassifier(n_jobs=-1, random_state=42))
])

# =========== XGBoost's Parameter for the case ===========
parameter = {
 'algo__max_depth': Integer(low=12, high=20),
 'algo__learning_rate': Real(low=-2, high=0, prior='log-uniform'),
 'algo__n_estimators': Integer(low=195, high=210),
 'algo__subsample': Real(low=0.5, high=1, prior='uniform'),
 'algo__gamma': Integer(low=1, high=10),
 'algo__colsample_bytree': Real(low=0.1, high=1, prior='uniform')
}

# =========== Build Model ===========
model = RandomizedSearchCV(pipeline, parameter, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=50)
model.fit(X_train, y_train)

# =========== Model Evalutation ===========
print(model.best_params_)

print(model.score(X_train, y_train), model.score(X_test, y_test))

pred = model.predict(X_test)
print(classification_report(y_test, pred))
print()
print('Accuracy: ', accuracy_score(y_test, pred))
print()

# =========== Confusion Matrix Visualization ===========
cm = confusion_matrix(y_test, pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)

ax.set_xlabel('aktual')
ax.set_ylabel('prediksi')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Hoax','Valid'])
ax.yaxis.set_ticklabels(['Hoax','Valid'])
print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
