# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 11:06:18 2017

@author: Firdauz_Fanani
"""

import csv,numpy,pandas,nltk
import sklearn
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk import sent_tokenize, word_tokenize, pos_tag
import time

from testing2 import string_test,label_string_test
import matplotlib.pyplot as plt
import cPickle
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.learning_curve import learning_curve

#%%
start_time = time.time()
         
#Baca data CSV dengan pandas
df = pandas.read_csv('C:\Users\user\Google Drive\Skripsi\data_skripsi2.csv', 
                     names=["Content", "Label"])

head = df.head()
print head

#View berdasar label
groupby= df.groupby('Label').describe()
print groupby

Con= df['Content']
Label = df['Label']

#print df
#print Con
#print Label[18]

#%%

#view banyaknya huruf
df['length'] = df['Content'].map(lambda text: len(text))
deksripsi_jumlah = df.length.describe()

#print df.head()
#print deksripsi_jumlah

#print list(df.Content[df.length > 1000])

#%%

#Plot bannyaknya huruf dibanding frequensi
#df.length.plot(bins=20, kind='hist')

#%%

#Mengganti Slang Word data train
def replace_semua(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text
    
slang_word = {'dpt':'dapat','krn':'karena','utk':'untuk','tp':'tapi','sy':'saya',
                    'sj':'saja','bln':'bulan','bgt':'banget','knp':'kenapa','tlg ' : 'tolong '
                    ,'dwpan' : 'depan','dwngan' : 'dengan','byk' : 'banyak','smoga' : 'semoga',
                    'kpn' : 'kapan','sdh' : 'sudah','kelwbihan' : 'kelebihan','smua':'semua',
                    'brani' : 'berani','jd':'jadi',' tlng ' : ' tolong ','jgn2' : 'jangan-jangan',
                    'bagusn' : 'bagusan','lmynlh' : 'lumayanlah',' ngk ' : ' enggak ',' bner ' : ' bener ',
                    ' blom ' : ' belum ',' pk ' : ' pakai ','maff' : 'maaf','splikasi' :'aplikasi','dnk' : 'dong',
                    'knp' : 'kenapa','gk' : 'gak','klu' : 'kalau','bli' : 'beli','bncmark' : 'benchmark','bsa' : 'bisa',
                    'awal2' : 'awal-awal','ttg' : 'tentang','udh' : 'udah','blm' : 'belum','msuk' : 'masuk',
                    'abg' : 'abang','klo' : 'kalau','brp' : 'berapa','kalo' : 'kalau','reales' : 'rilis','blon' : 'belum',
                    'gmn' : 'gimana','gmna' : 'gimana','jwb' : 'jawab','yng' : 'yang','yg' : 'yang','syng' : 'sayang','bgs' : 'bagus',
                    'idup' : 'hidup','dgn' : 'dengan','dpn' : 'depan','dpnx' : 'depannya','blkng' : 'belakang','ngak' : 'enggak','dripada' : 'daripada','bgt' : 'banget','bda' : 'beda','pke' : 'pake','kpke' : 'kepakai','bw' : 'bawa','bs' : 'bisa','knapa' : 'kenapa','hpx' : 'hapenya','bgus2' : 'bagus-bagus','bls' : 'balas','lg' : 'lagi','ttp' : 'tetap','km' : 'kamu','dr' : 'dari','tmn2' : 'teman-teman','blakang' : 'belakang','dri' : 'dari','skrg' : 'sekarang','jg' : 'juga','gpp' : 'gapapa','wktu' : 'waktu','tmpt' : 'tempat','dibles' : 'dibales','karna' : 'karena','stelan' : 'setelan','kwalitas' : 'kualitas','tdk' : 'tidak','jga' : 'juga','mnta' : 'minta','lbih' : 'lebih','riwue' : 'review','skligus':'sekaligus','Ccok':'cocok','drpd':'daripada','tdk':'tidak','trs':'terus','pdhl':'padahal','tlp':'telepon','telefon':'telepon','udh':'udah','dmn':'dimana','smg':'semoga','sblm':'sebelum','lbh':'lebih','tmpt':'tempat','byr':'bayar','thn':'tahun','tsb':'tersebut','spt':'seperti','spy':'supaya'}

slang=[]

for conten in df['Content']:
    txt = replace_semua(conten.lower(), slang_word)
    slang.append(txt)

df['slang'] = slang

#print slang

#%%

#Stop word data train

stopword = ['ada','adalah','adanya','adapun','agak','agaknya','agar','akan','akankah',
                 'akhir','akhiri','akhirnya','aku','akulah','amat','amatlah','anda','andalah',
                 'antar','antara','antaranya','apa','apaan','apabila','apakah','apalagi','apatah',
                 'artinya','asal','asalkan','atas','atau','ataukah','ataupun','awal','awalnya','bagai',
                 'bagaikan','bagaimana','bagaimanakah','bagaimanapun','bagi','bagian','bahkan','bahwa',
                 'bahwasanya','baik','bakal','bakalan','balik','banyak','bapak','baru','bawah','beberapa',
                 'begini','beginian','beginikah','beginilah','begitu','begitukah','begitulah','begitupun',
                 'bekerja','belakang','belakangan','belum','belumlah','benar','benarkah','benarlah','berada',
                 'berakhir','berakhirlah','berakhirnya','berapa','berapakah','berapalah','berapapun','berarti',
                 'berawal','berbagai','berdatangan','beri','berikan','berikut','berikutnya','berjumlah','berkali-kali',
                 'berkata','berkehendak','berkeinginan','berkenaan','berlainan','berlalu','berlangsung','berlebihan','bermacam',
                 'bermacam-macam','bermaksud','bermula','bersama','bersama-sama','bersiap','bersiap-siap','bertanya','bertanya-tanya','berturut','berturut-turut','bertutur','berujar','berupa','besar','betul','betulkah','biasa','biasanya','bila','bilakah','bisa','bisakah','boleh','bolehkah','bolehlah','buat','bukan','bukankah','bukanlah','bukannya','bulan','bung','cara','caranya','cukup','cukupkah','cukuplah','cuma','dahulu','dalam','dan','dapat','dari','daripada','datang','dekat','demi','demikian','demikianlah','dengan','depan','di','dia','diakhiri','diakhirinya','dialah','diantara','diantaranya','diberi','diberikan','diberikannya','dibuat','dibuatnya','didapat','didatangkan','digunakan','diibaratkan','diibaratkannya','diingat','diingatkan','diinginkan','dijawab','dijelaskan','dijelaskannya','dikarenakan','dikatakan','dikatakannya','dikerjakan','diketahui','diketahuinya','dikira','dilakukan','dilalui','dilihat','dimaksud','dimaksudkan','dimaksudkannya','dimaksudnya','diminta','dimintai','dimisalkan','dimulai','dimulailah','dimulainya','dimungkinkan','dini','dipastikan','diperbuat','diperbuatnya','dipergunakan','diperkirakan','diperlihatkan','diperlukan','diperlukannya','dipersoalkan','dipertanyakan','dipunyai','diri','dirinya','disampaikan','disebut','disebutkan','disebutkannya','disini','disinilah','ditambahkan','ditandaskan','ditanya','ditanyai','ditanyakan','ditegaskan','ditujukan','ditunjuk','ditunjuki','ditunjukkan','ditunjukkannya','ditunjuknya','dituturkan','dituturkannya','diucapkan','diucapkannya','diungkapkan','dong','dua','dulu','empat','enggak','enggaknya','entah','entahlah','guna','gunakan','hal','hampir','hanya','hanyalah','hari','harus','haruslah','harusnya','hendak','hendaklah','hendaknya','hingga','ia','ialah','ibarat','ibaratkan','ibaratnya','ibu','ikut','ingat','ingat-ingat','ingin','inginkah','inginkan','ini','inikah','inilah','itu','itukah','itulah','jadi','jadilah','jadinya','jangan','jangankan','janganlah','jauh','jawab','jawaban','jawabnya','jelas','jelaskan','jelaslah','jelasnya','jika','jikalau','juga','jumlah','jumlahnya','justru','kala','kalau','kalaulah','kalaupun','kalian','kami','kamilah','kamu','kamulah','kan','kapan','kapankah','kapanpun','karena','karenanya','kasus','kata','katakan','katakanlah','katanya','ke','keadaan','kebetulan','kecil','kedua','keduanya','keinginan','kelamaan','kelihatan','kelihatannya','kelima','keluar','kembali','kemudian','kemungkinan','kemungkinannya','kenapa','kepada','kepadanya','kesampaian','keseluruhan','keseluruhannya','keterlaluan','ketika','khususnya','kini','kinilah','kira','kira-kira','kiranya','kita','kitalah','kok','kurang','lagi','lagian','lah','lain','lainnya','lalu','lama','lamanya','lanjut','lanjutnya','lebih','lewat','lima','luar','macam','maka','makanya','makin','malah','malahan','mampu','mampukah','mana','manakala','manalagi','masa','masalah','masalahnya','masih','masihkah','masing','masing-masing','mau','maupun','melainkan','melakukan','melalui','melihat','melihatnya','memang','memastikan','memberi','memberikan','membuat','memerlukan','memihak','meminta','memintakan','memisalkan','memperbuat','mempergunakan','memperkirakan','memperlihatkan','mempersiapkan','mempersoalkan','mempertanyakan','mempunyai','memulai','memungkinkan','menaiki','menambahkan','menandaskan','menanti','menanti-nanti','menantikan','menanya','menanyai','menanyakan','mendapat','mendapatkan','mendatang','mendatangi','mendatangkan','menegaskan','mengakhiri','mengapa','mengatakan','mengatakannya','mengenai','mengerjakan','mengetahui','menggunakan','menghendaki','mengibaratkan','mengibaratkannya','mengingat','mengingatkan','menginginkan','mengira','mengucapkan','mengucapkannya','mengungkapkan','menjadi','menjawab','menjelaskan','menuju','menunjuk','menunjuki','menunjukkan','menunjuknya','menurut','menuturkan','menyampaikan','menyangkut','menyatakan','menyebutkan','menyeluruh','menyiapkan','merasa','mereka','merekalah','merupakan','meski','meskipun','meyakini','meyakinkan','minta','mirip','misal','misalkan','misalnya','mula','mulai','mulailah','mulanya','mungkin','mungkinkah','nah','naik','namun','nanti','nantinya','nyaris','nyatanya','oleh','olehnya','pada','padahal','padanya','pak','paling','panjang','pantas','para','pasti','pastilah','penting','pentingnya','per','percuma','perlu','perlukah','perlunya','pernah','persoalan','pertama','pertama-tama','pertanyaan','pertanyakan','pihak','pihaknya','pukul','pula','pun','punya','rasa','rasanya','rata','rupanya','saat','saatnya','saja','sajalah','saling','sama','sama-sama','sambil','sampai','sampai-sampai','sampaikan','sana','sangat','sangatlah','satu','saya','sayalah','se','sebab','sebabnya','sebagai','sebagaimana','sebagainya','sebagian','sebaik','sebaik-baiknya','sebaiknya','sebaliknya','sebanyak','sebegini','sebegitu','sebelum','sebelumnya','sebenarnya','seberapa','sebesar','sebetulnya','sebisanya','sebuah','sebut','sebutlah','sebutnya','secara','secukupnya','sedang','sedangkan','sedemikian','sedikit','sedikitnya','seenaknya','segala','segalanya','segera','seharusnya','sehingga','seingat','sejak','sejauh','sejenak','sejumlah','sekadar','sekadarnya','sekali','sekali-kali','sekalian','sekaligus','sekalipun','sekarang','sekarang','sekecil','seketika','sekiranya','sekitar','sekitarnya','sekurang-kurangnya','sekurangnya','sela','selain','selaku','selalu','selama','selama-lamanya','selamanya','selanjutnya','seluruh','seluruhnya','semacam','semakin','semampu','semampunya','semasa','semasih','semata','semata-mata','semaunya','sementara','semisal','semisalnya','sempat','semua','semuanya','semula','sendiri','sendirian','sendirinya','seolah','seolah-olah','seorang','sepanjang','sepantasnya','sepantasnyalah','seperlunya','seperti','sepertinya','sepihak','sering','seringnya','serta','serupa','sesaat','sesama','sesampai','sesegera','sesekali','seseorang','sesuatu','sesuatunya','sesudah','sesudahnya','setelah','setempat','setengah','seterusnya','setiap','setiba','setibanya','setidak-tidaknya','setidaknya','setinggi','seusai','sewaktu','siap','siapa','siapakah','siapapun','sini','sinilah','soal','soalnya','suatu','sudah','sudahkah','sudahlah','supaya','tadi','tadinya','tahu','tahun','tak','tambah','tambahnya','tampak','tampaknya','tandas','tandasnya','tanpa','tanya','tanyakan','tanyanya','tapi','tegas','tegasnya','telah','tempat','tengah','tentang','tentu','tentulah','tentunya','tepat','terakhir','terasa','terbanyak','terdahulu','terdapat','terdiri','terhadap','terhadapnya','teringat','teringat-ingat','terjadi','terjadilah','terjadinya','terkira','terlalu','terlebih','terlihat','termasuk','ternyata','tersampaikan','tersebut','tersebutlah','tertentu','tertuju','terus','terutama','tetap','tetapi','tiap','tiba','tiba-tiba','tidak','tidakkah','tidaklah','tiga','tinggi','toh','tunjuk','turut','tutur','tuturnya','ucap','ucapnya','ujar','ujarnya','umum','umumnya','ungkap','ungkapnya','untuk','usah','usai','waduh','wah','wahai','waktu','waktunya','walau','walaupun','wong','yaitu','yakin','yakni','yang']

sw=[]

for conten in slang:
    kata= filter(lambda x: x not in stopword, conten)
    sw.append(kata)

df['stopword'] = sw


#%%
# buat stemmer data train
start_time = time.time()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stem
st=[]

for conten in sw:
    output   = stemmer.stem(conten)
    st.append(output)

df['stemmed'] = st

#print st
print "My program took", time.time() - start_time, "to run"
#%%

#Pisahkan kata/tokenize data train

PK= [nltk.word_tokenize(PisahKata) for PisahKata in st]

#print PK
#print sw

#panjang = len(kata)
#print panjang

#%%

#Ubah list ke str
string = map(' '.join, PK)

#print string
print "My program took", time.time() - start_time, "to run"
#%%

start_time = time.time()
#Count Vector
count_vect = CountVectorizer()
count = count_vect.fit_transform(string)
XtrainCount = count.toarray()

##print len(XtrainCount)
#print len(count_vect.vocabulary_)
#print count_vect.get_feature_names()

#bag of words

print count
#print count.shape[0]
#print count.shape[1]
print 'sparse matrix shape:', count.shape
print 'number of non-zeros:', count.nnz
print 'sparsity: %.2f%%' % (100.0 * count.nnz / (count.shape[0] * count.shape[1]))

#Normalisasi tfdf
tf_transformer = TfidfTransformer().fit(count)
X_train_tf = tf_transformer.transform(count)
Xcount = X_train_tf.todense()

print tf_transformer
print X_train_tf
print Xcount
Xcoba= X_train_tf.astype('str')

print "My program took", time.time() - start_time, "to run"

#%%

start_time = time.time()
#Predictor Naive Bayes BernoulliNB() & SVM LinearSVC()
spam_detector = LinearSVC().fit(count, Label)

#print 'predicted:', spam_detector.predict(X_train_tf)[19]
#print 'expected:', Label[3]
#print X_train_tf[19]
#print Label[3]

prediksi_semua = spam_detector.predict(count)

print prediksi_semua

print 'accuracy', accuracy_score(Label, prediksi_semua)
print 'confusion matrix\n', confusion_matrix(Label, prediksi_semua)
print '(row=expected, col=predicted)'

print classification_report(Label, prediksi_semua)

print "My program took", time.time() - start_time, "to run"
'''
def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_spam_coefficients = numpy.argsort(coef)[-top_features:]
 top_ham_coefficients = numpy.argsort(coef)[:top_features]
 top_coefficients = numpy.hstack([top_ham_coefficients, top_spam_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ["blue" if c < 0 else "red" for c in coef[top_coefficients]]
 plt.bar(numpy.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = numpy.array(feature_names)
 plt.xticks(numpy.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
 plt.show()

svm = LinearSVC()
svm.fit(X_train_tf, Label)

plot_coefficients(svm, count_vect.get_feature_names())
'''



'''
plt.matshow(confusion_matrix(Label, prediksi_semua), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
'''

#%%

#Performing klasifikasi dgn cross validation
#Naive Bayes BernoulliNB() & SVM LinearSVC()
start_time = time.time()
msg_train, msg_test, label_train, label_test = \
    train_test_split(string, Label, test_size=0.1)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

klasifikasi_text = Pipeline([
    ('vectorizer', CountVectorizer()),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', BernoulliNB()),])
    
print klasifikasi_text

scores = cross_val_score(klasifikasi_text,  
                         string,  
                         Label,  
                         cv=StratifiedKFold(Label, n_folds=10),    
                         scoring='accuracy',  
                         n_jobs=-1,
                         )
                         
print scores

print scores.mean(), scores.std()

print "My program took", time.time() - start_time, "to run"

'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=numpy.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Jumlah Data Latih")
    plt.ylabel("Akurasi")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Hasil Klasifikasi dengan Data Sendiri")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Hasil Klasifikasi dengan Cross-validation")

    plt.legend(loc="best")
    return plt
print plot_learning_curve(klasifikasi_text, "Akurasi vs. Jumlah Data Latih", string, Label, cv=10)
'''
#%%

start_time = time.time()

#Grid Search untuk penyesuaian parameter 

params = {
    'tfidf__use_idf': (True, False),
}

grid = GridSearchCV(
    klasifikasi_text,  
    params,  
    refit=True,  
    n_jobs=-1,  
    scoring='accuracy',     
)

print grid

spaam_detector = grid.fit(string,Label)

print spaam_detector.grid_scores_

#print spaam_detector.predict_proba(string_test)
#print spaam_detector.predict(["hahaha","Maaf...ya...terpaksa sy kurangi dua bintang...karena belum bisa di gunakan untuk membeli token listrik...klo sudah bisa nanti akan saya tambah bintang 5..."])

predictions = spaam_detector.predict(string_test)
print 'accuracy', accuracy_score(label_string_test, predictions)
print 'confusion matrix\n', confusion_matrix(label_string_test, predictions)
print classification_report(label_string_test, predictions)

predik= predictions.astype('str')
#prob = spaam_detector.predict_proba(string_test)
print predictions

print "My program took", time.time() - start_time, "to run"

