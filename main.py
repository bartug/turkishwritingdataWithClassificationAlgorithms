#Berke Bartuğ Sevindik
import re
import time
import string
import numpy as np
import pandas as pd
from collections import Counter
from zemberek import TurkishMorphology

# önişleme için gerekli veriler dosyadan alınıyor.
with open('turkce_veri_dosyalari/stop-words-turkish-github.txt', mode='r', encoding='windows-1254') as f:
    STOPWORDS = f.read().splitlines()

with open('turkce_veri_dosyalari/turkce_isimler-github.txt', encoding='utf-8') as f:
    turkce_isimler = f.read().splitlines()

remove_digits = str.maketrans(' ', ' ', string.digits)
remove_punc = str.maketrans(' ', ' ', string.punctuation)
tm = TurkishMorphology.create_with_defaults()

dataset = pd.read_csv('texts.csv', index_col=(['index']))
print(len(dataset))
print(dataset)
dataset['label'] = None
dataset['label'][:3422] = pd.Series(['Turkish 101'] * 3422)
dataset['label'][3422:] = pd.Series(['Turkish 102'] * 3422) #6883
dataset.dropna(inplace=True)

# ilk ve  son 50 satır ile denemeler yapılması için
# veriseti gerçek verisetinin bu kısmından oluşuyor.
#dataset = pd.concat([dataset.iloc[:100], dataset.iloc[6742:]])  # sşil
y = dataset['label']


# TurkishMorphology sınıfının kullanarak alınan sözcüğün kökünü bulan fonksiyon
def get_root(word: str):
    analysis = tm.analyze(word).analysis_results
    try:
        return analysis[1].item.root
    except IndexError:
        try:
            return analysis[0].item.root
        except IndexError:
            return word


# Parametre olarak alınan metinden numerik karakterleri, noktalama işaretleri, stop-wordsleri ve
# özel isimleri çıkartan ve geri kalan sözcüklerin kökünü bulan fonksiyon
def text_to_words(text):
    text = text.translate(remove_digits)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = [word.lower() for word in text.split() if not word in turkce_isimler]
    text = [word for word in text if not word in STOPWORDS]
    text = [get_root(word) for word in text]
    return text


# Her bir sözcüğün hangi metinde kaç defa geçtiği bulunuyor.
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer=text_to_words, ngram_range=(1, 1))
cv.fit(dataset['text'])
# print(cv.vocabulary_)
print(len(cv.vocabulary_)) #farklı kelime sayisi

sparse = cv.transform(dataset['text'])
print(sparse.shape)

# tf-idf değerleri hesaplatılıyor.
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(sparse)
sparse = tfidf_transformer.transform(sparse)

# Ki-Kare feature selection en cok anlam ifade edenler
from sklearn.feature_selection import SelectKBest, chi2

k = 5000  # kelimeyi düşürüyorum
ch2 = SelectKBest(chi2, k=k)

sparse = ch2.fit_transform(sparse, y)
print(sparse.shape)

# train ve test kümeleri bölünüyor.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sparse, y, test_size=0.25, random_state=0)

                #Siniflandirma Algoritmalaril
# 1) Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# modelin performans ölçüm değerleri hesaplatılıyor.
from sklearn.metrics import confusion_matrix, classification_report
print("\nMultinomial Naive Bayes:")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))


# 2) K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15, metric='cosine')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# modelin performans ölçüm değerleri hesaplatılıyor.
print("\nK-Nearest Neighbors:")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))


#3) Support Vector Machines
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
parameters = {'loss':['hinge'],
              'alpha':[0.1,0.01,0.001,0.0001],
              'max_iter':[100,250,500,1000,2000,3000],
              }
gs_clf = GridSearchCV(SGDClassifier(), param_grid=parameters, cv=10, n_jobs=-1)
gs_clf.fit(X_train, y_train)
y_pred = gs_clf.predict(X_test)

# modelin performans ölçüm değerleri hesaplatılıyor.
print("\nSupport Vector Machines:")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# 4) LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# modelin performans ölçüm değerleri hesaplatılıyor.
print("\nLogistic Regression:")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# 5) Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500, criterion='gini')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# modelin performans ölçüm değerleri hesaplatılıyor.
print("\nRandom Forest:")
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))