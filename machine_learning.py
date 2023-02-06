import pandas as pd
#import re
import nltk
#from nltk.stem.porter import PorterStemmer
import codecs
from nltk import sent_tokenize
nltk.download('averaged_perceptron_tagger')

"""
nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
#stoplist.remove ("not")
#stoplist.remove ("no")
#stoplist.remove ("nor")
#stoplist.remove ("all")
#stoplist.remove ("just")
ps = PorterStemmer()
"""

dataset = pd.read_csv("data.csv", sep=";", quoting=3)
reviews = dataset['Texte'].values
#print(dataset)
X = dataset.iloc[:, 2:-1].values
#print(X)


cats = ["VBZ", "VB", "NNP", "RBR", "RBS", "JJS", "JJR","RB","RP","VBD","VBG","VBN","VBP", "WRB"] #"UH" ,"DT","MD","JJ","PDT","NN", ,
corpus = []
for Texte in reviews:
    sentences = sent_tokenize(Texte)
    words = nltk.word_tokenize(Texte)
    mots = []
    testE = nltk.pos_tag(words)
    for w in testE:
        if w[1] in cats:
            mots.append(w[0])
    Texte = Texte.lower()
    Texte = " ".join(mots)
    #print(testE)
    corpus.append(Texte)
#print(corpus)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=4000) #variable 
X = cv.fit_transform(corpus).toarray() #applique cv à corpus

print (X, len(X[0]) )
print(cv.vocabulary_)
print(cv.get_feature_names() )

y = dataset.iloc[:, 1].values #prend seulement la valeur de la colonne
#print(y)

from sklearn .model_selection import train_test_split
X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)
                                                    
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


for i in range(len(X_test)):
    print("valeur réelle", y_test[i], "prédiction", y_pred[i])
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


with codecs.open("random_feedbacks.txt","r","utf8") as f:
    avis = f.readlines()
def pre_traitement(text): 
    #sentences = sent_tokenize(text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    mots = []
    testE = nltk.pos_tag(words)

    for w in testE:
        if w[1] in cats:
            mots.append(w[0])
    
    Texte = " ".join(mots)
    Texte = [Texte]
    Texte = cv.transform(Texte).toarray()
    y_new = classifier.predict(Texte)

    #print(testE)
    for i in range(len(Texte[0])):
        for w in cv.vocabulary_:
            if i == cv.vocabulary_[w] and Texte[0][i] != 0:
                print(w,", fréquence du mot:",Texte[0][i])
    if y_new[0] == 'pos':
        return "positif"
    else:
        return "négatif"
#print(corpus)
for x in avis:
    print(x, ":", pre_traitement(x))