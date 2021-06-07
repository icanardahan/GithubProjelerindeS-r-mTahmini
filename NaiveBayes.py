#Temizlenmiş verinin yüklenmesi
import pandas as pd

data = pd.read_excel('Release_Notes_v0_cleaned.xlsx')
print(data.head())

#Veri Setini Bölme 

#Temizlenmiş veriyi train ve test kümelerine ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['Note'].values.astype('U'),
                                                    data['Type'].values.astype('U'), test_size = 0.1, random_state = 42)

print(X_train.shape)
print(X_test.shape)

#Sayma Vektörü Oluşturma
#Train kümesindeki cümlelerin sayma vektörlerini çıkarıyoruz
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform((X_train))
print(X_train_counts.shape)

#Train kümesindeki cümlelerin TF*IDF vektörlerini sayma vektörlerinden oluşturuyoruz
#TF-IDF, bir doküman uzayi içerisinde geçen bir kelimenin herhangi bir doküman içerisinde ne kadar önemli olduğunu belirtmek için tasarlanmış istatiksel bir ölçüttür.

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

#Naive Bayes Model Eğitimi
#Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#Model Performansı Ölçme

#Sınıflandırıcı ile test seti üzerinde tahminleme yapılması
y_pred = clf.predict(X_test_tfidf)
for Note, sentiment in zip(X_test[:10], y_pred[:]):
    print ('%r => %s' % (Note, sentiment)) 
    
#Test Sonuçları
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))    