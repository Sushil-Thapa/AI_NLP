import sys
import os
import numpy as np
import pandas as pd


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):

    pos_files = os.listdir(inpath+"/pos")
    neg_files = os.listdir(inpath+"/neg")
    outfile = open(outpath+name,"w")
    #outfile.write(',text,polarity\n')
    data = pd.DataFrame(columns=('text','polarity'))
    data.to_csv(outfile)
    count = 0
    for files in pos_files:
            with open(inpath+"/pos/"+files,'r') as f:
                    row = pd.DataFrame([[f.read(),1]],columns=['text','polarity'],index=[count])
                    row.to_csv(outfile,header=False)
                    count += 1
    for files in neg_files:
            with open(inpath+"/neg/"+files,'r') as f:
                    row = pd.DataFrame([[f.read(),0]],columns=['text','polarity'],index=[count])
                    #data.append(row)
                    count += 1
                    row.to_csv(outfile,header=False)


if __name__ == "__main__":
    data =  pd.read_csv("./imdb_tr.csv",encoding="ISO-8859-1",index_col=0)
    train, val, train_target, val_target = train_test_split(data['text'].values,data['polarity'].values,test_size = 0.2,random_state=0)

    with open('stopwords.en.txt') as f:
            stop_word = f.read().split("\n")
    test_file = 'imdb_tr.csv'
    test = pd.read_csv(test_file,encoding="ISO-8859-1",index_col=0)

    unigram = CountVectorizer(stop_words=stop_word)
    unigram_train = unigram.fit_transform(train)
    unigram_val = unigram.transform(val)
    unigram_test = unigram.transform(test['text'].values)

    params = [{'alpha':[0.0001,0.001,0.01,0.02,0.05,0.01],'penalty':['l1','l2']}]

    sgd = SGDClassifier(loss="hinge")
    clf = GridSearchCV(estimator=sgd,param_grid=params,n_jobs=-1,cv=5)
    clf.fit(unigram_train,train_target)
    train_pre = clf.predict(unigram_train)
    print(accuracy_score(train_target,train_pre))
    val_pre = clf.predict(unigram_val)
    print(accuracy_score(val_target,val_pre))
    test_pre = clf.predict(unigram_test)
    np.savetxt('unigram.output.txt',test_pre,fmt="%d",delimiter="/n")

    unigramT = TfidfVectorizer(stop_words=stop_word)
    unigramT_train = unigramT.fit_transform(train)
    unigramT_val = unigramT.transform(val)
    unigramT_test = unigramT.transform(test['text'].values)

    clf.fit(unigramT_train,train_target)
    train_pre = clf.predict(unigramT_train)
    print(accuracy_score(train_target,train_pre))
    val_pre = clf.predict(unigramT_val)
    print(accuracy_score(val_target,val_pre))
    test_pre = clf.predict(unigramT_test)
    np.savetxt('unigramtfidf.output.txt',test_pre,fmt="%d",delimiter="/n")

    bigram = CountVectorizer(stop_words=stop_word,ngram_range=(1,2))
    bigram_train = bigram.fit_transform(train)
    bigram_val = bigram.transform(val)
    bigram_test = bigram.transform(test['text'].values)

    params = [{'alpha':[0.0001,0.001,0.01,0.02,0.05,0.01],'penalty':['l1','l2']}]

    clf.fit(bigram_train,train_target)
    train_pre = clf.predict(bigram_train)
    print(accuracy_score(train_target,train_pre))
    val_pre = clf.predict(bigram_val)
    print(accuracy_score(val_target,val_pre))
    test_pre = clf.predict(bigram_test)
    np.savetxt('bigram.output.txt',test_pre,fmt="%d",delimiter="/n")

    bigramT = TfidfVectorizer(stop_words=stop_word,ngram_range=(1,2))
    bigramT_train = bigramT.fit_transform(train)
    bigramT_val = bigramT.transform(val)
    bigramT_test = bigramT.transform(test['text'].values)

    clf.fit(bigramT_train,train_target)
    train_pre = clf.predict(bigramT_train)
    print(accuracy_score(train_target,train_pre))
    val_pre = clf.predict(bigramT_val)
    print(accuracy_score(val_target,val_pre))
    test_pre = clf.predict(bigramT_test)
    np.savetxt('bigramtfidf.output.txt',test_pre,fmt="%d",delimiter="/n")
