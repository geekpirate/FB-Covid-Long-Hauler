from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import roc_curve
from gensim.models import Word2Vec
import expROC
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.backend import clear_session
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tensorflow.python.keras import Sequential
import tensorflow as tf
import word2vec
from keras.preprocessing.text import Tokenizer
import xlrd
import xlwt
from keras import backend as K
from sklearn.model_selection import cross_val_score
from liwc import Liwc
from liwc_features import LIWCExtractor
from matplotlib import pyplot


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)


    if is_neural_net:
        return classifier.evaluate(feature_vector_valid, valid_y, batch_size=10, verbose=1)
        #predictions = predictions.argmax(axis=-1)
        #predictions = numpy.argmax(predictions, axis = -1)
        #print("Score =", classifier.evaluate(feature_vector_valid, valid_y, batch_size=10, verbose=1))
        #print(metrics.classification_report(valid_y, predictions))
        #print(predictions, valid_y)
        #exit()
    clear_session()
    acc.append(metrics.accuracy_score(valid_y, predictions))
    prec.append(metrics.precision_score(valid_y, predictions))
    rec.append(metrics.recall_score(valid_y, predictions))
    F1.append(metrics.f1_score(valid_y, predictions))
    AUC.append(metrics.roc_auc_score(valid_y, predictions))
    return metrics.accuracy_score(valid_y, predictions), metrics.precision_score(valid_y, predictions), \
           metrics.recall_score(valid_y, predictions), metrics.f1_score(valid_y, predictions), \
           metrics.roc_auc_score(valid_y, predictions)
acc = []
prec = []
rec = []
F1 = []
AUC = []
def printInputData(texts, labels):

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    # create a dataframe using texts and lables

    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    skf.get_n_splits(trainDF['text'], trainDF['label'])

    X = trainDF['text']
    y = trainDF['label']
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        train_x, valid_x = X[train_index], X[test_index]
        train_y, valid_y = y[train_index], y[test_index]
        #word2vec
        # split the dataset into training and validation datasets
        #train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],
        #train_size=0.8)
        #X = numpy.array(trainDF['text'])
        #y= numpy.array(trainDF['label'])
        #for train_index, test_index in skf.split(X, y):
            #print("TRAIN:", train_index, "TEST:", test_index)
            #train_x, valid_x = X[train_index], X[test_index]
            #train_y, valid_y = y[train_index], y[test_index]

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)

        # Count Vectors as features
        liwc_features = []
        wb = xlrd.open_workbook("C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/LIWC2015 Results (test_liwc).xls")
        sh = wb.sheet_by_name("Sheet0")
        for i in range(1,sh.nrows):
            temp = []
            for j in range(1,94):
                temp.append(sh.cell_value(i, j))
            liwc_features.append(temp)
        #features = features.toarray()
        #scaler = preprocessing.MinMaxScaler()
        #features = scaler.fit_transform(features)
        #exit()

        vectorizer = FeatureUnion([
            ('count_vect', CountVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
            ('tfidf_vect', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
            ('tfidf_vect_ngram', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}',
                                                 ngram_range=(1, 4))),
            ('tfidf_vect_ngram_chars', TfidfVectorizer(stop_words='english', analyzer='char', token_pattern=r'\w{1,}',
                                                       ngram_range=(1, 4)))
        ])
        vectorizer.fit(trainDF['text'])
        train_features = vectorizer.transform(train_x)
        test_features = vectorizer.transform(valid_x)

        train_features_updated = []
        test_features_updated = []
        for prev in range(len(train_features.toarray())):
            temp = []
            for i in train_features.toarray()[prev]:
                temp.append(i)
            train_features_updated.append(temp + liwc_features[prev])
        train_features_updated = numpy.array(train_features_updated)
        for prev in range(len(test_features.toarray())):
            temp = []
            for i in test_features.toarray()[prev]:
                temp.append(i)

            test_features_updated.append(temp + liwc_features[prev + len(train_features.toarray()) - 1])
        test_features_updated = numpy.array(test_features_updated)

        # 'Xtereme Gradient Boosting Model
        print("Xtereme Gradient Boosting Model------------")
        # Extereme Gradient Boosting on Count Vectors
        accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features_updated, train_y, test_features_updated,
                                        valid_y, is_neural_net=False)
        print("LIWC Xgb,: ", accuracy_GB_Count)

    print("ACC", numpy.min(acc), numpy.max(acc), numpy.mean(acc), numpy.std(acc))
    print("Prec", numpy.min(prec), numpy.max(prec), numpy.mean(prec), numpy.std(prec))
    print("Rec", numpy.min(rec), numpy.max(rec), numpy.mean(rec), numpy.std(rec))
    print("F1", numpy.min(F1), numpy.max(F1), numpy.mean(F1), numpy.std(F1))
    print("AUC", numpy.min(AUC), numpy.max(AUC), numpy.mean(AUC), numpy.std(AUC))
    exit()

    vectorizer = FeatureUnion([
        ('count_vect', CountVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}'))
    ])
    vectorizer.fit(trainDF['text'])
    train_features = vectorizer.transform(train_x)
    test_features = vectorizer.transform(valid_x)

    # 'Xtereme Gradient Boosting Model
    print("Xtereme Gradient Boosting Model------------")
    # Extereme Gradient Boosting on Count Vectors
    accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features, train_y, test_features,
                                    valid_y, is_neural_net=False)
    print("Count Xgb,: ", accuracy_GB_Count)

    vectorizer = FeatureUnion([
        ('count_vect', CountVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}'))
    ])
    vectorizer.fit(trainDF['text'])
    train_features = vectorizer.transform(train_x)
    test_features = vectorizer.transform(valid_x)

    # 'Xtereme Gradient Boosting Model
    print("Xtereme Gradient Boosting Model------------")
    # Extereme Gradient Boosting on Count Vectors
    accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features, train_y, test_features,
                                    valid_y, is_neural_net=False)
    print("TFIDF Xgb,: ", accuracy_GB_Count)

    vectorizer = FeatureUnion([
        ('count_vect', CountVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect_ngram', TfidfVectorizer(stop_words='english', analyzer='char', token_pattern=r'\w{1,}',
                                             ngram_range=(1, 4))),
    ])
    vectorizer.fit(trainDF['text'])
    train_features = vectorizer.transform(train_x)
    test_features = vectorizer.transform(valid_x)

    # 'Xtereme Gradient Boosting Model
    print("Xtereme Gradient Boosting Model------------")
    # Extereme Gradient Boosting on Count Vectors
    accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features, train_y, test_features,
                                    valid_y, is_neural_net=False)
    print("n-gram Xgb,: ", accuracy_GB_Count)

    vectorizer = FeatureUnion([
        ('count_vect', CountVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect_ngram', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}',
                                             ngram_range=(1, 4))),
        ('tfidf_vect_ngram_chars', TfidfVectorizer(stop_words='english', analyzer='char', token_pattern=r'\w{1,}',
                                            ngram_range=(1, 4)))
    ])
    vectorizer.fit(trainDF['text'])
    train_features = vectorizer.transform(train_x)
    test_features = vectorizer.transform(valid_x)

    # 'Xtereme Gradient Boosting Model
    print("Xtereme Gradient Boosting Model------------")
    # Extereme Gradient Boosting on Count Vectors
    accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features, train_y, test_features,
                                    valid_y, is_neural_net=False)
    print("word2vec Xgb,: ", accuracy_GB_Count)

    vectorizer = FeatureUnion([
        ('count_vect', CountVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}')),
        ('tfidf_vect_ngram', TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=r'\w{1,}',
                                             ngram_range=(1, 4))),
        ('tfidf_vect_ngram_chars', TfidfVectorizer(stop_words='english', analyzer='char', token_pattern=r'\w{1,}',
                                                   ngram_range=(1, 4)))
    ])
    vectorizer.fit(trainDF['text'])
    train_features = vectorizer.transform(train_x)
    test_features = vectorizer.transform(valid_x)

    train_features_updated = []
    test_features_updated = []
    for prev in range(len(train_features.toarray())):
        temp = []
        for i in train_features.toarray()[prev]:
            temp.append(i)
        train_features_updated.append(temp + liwc_features[prev])
    train_features_updated = numpy.array(train_features_updated)
    for prev in range(len(test_features.toarray())):
        temp = []
        for i in test_features.toarray()[prev]:
            temp.append(i)

        test_features_updated.append(temp + liwc_features[prev+len(train_features.toarray())-1])
    test_features_updated = numpy.array(test_features_updated)

    # 'Xtereme Gradient Boosting Model
    print("Xtereme Gradient Boosting Model------------")
    # Extereme Gradient Boosting on Count Vectors
    accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features_updated, train_y, test_features_updated,
                                    valid_y, is_neural_net=False)
    print("LIWC Xgb,: ", accuracy_GB_Count)

    exit()
    #print(train_features.toarray() + features[:462])
    MAX_NUM_WORDS = 95000

    # SVM
    # SVM on Ngram Level TF IDF Vectors
    print("SVM ---------------------------")
    accuracy_SVM_Count = train_model(svm.SVC(), train_features_updated, train_y, test_features_updated, valid_y,
                                     is_neural_net=False)
    print("SVM, Count Vectors: ", accuracy_SVM_Count)

# Naive Bayes

    # Naive Bayes on Count Vectors
    print("Naive Bayes: ----------------")

    accuracy_NB_Count = train_model(naive_bayes.MultinomialNB(), train_features_updated, train_y,
                                    test_features_updated,
                                    valid_y, is_neural_net=False)
    print("NB, Count Vectors: ", accuracy_NB_Count)

# Linear Classifier
    print("Linear CLassifier --------------------")
    # Linear Classifier on Count Vectors
    accuracy_LR_Count = train_model(linear_model.LogisticRegression(), train_features_updated, train_y, test_features_updated,
                                    valid_y, is_neural_net=False)
    print("LR, Count Vectors: ", accuracy_LR_Count)




#Bagging Model - Random Forest
    print("Bagging Model - Random Forest----------- ")
    # RF on Count Vectors
    accuracy_RF_Count = train_model(ensemble.RandomForestClassifier(), train_features_updated, train_y, test_features_updated,
                                    valid_y, is_neural_net=False)
    print("RF, Count Vectors: ", accuracy_RF_Count)

# 'Xtereme Gradient Boosting Model
    print("Xtereme Gradient Boosting Model------------")
    # Extereme Gradient Boosting on Count Vectors
    accuracy_GB_Count = train_model(xgboost.XGBClassifier(), train_features_updated, train_y, test_features_updated,
                                    valid_y, is_neural_net=False)
    print("Xgb,: ", accuracy_GB_Count)
    pyplot.show()
    exit()
#Shallow Neural Networks

    def create_model_architecture(input_size):
        # create input layer
        input_layer = layers.Input((input_size,), sparse=True)

        # create hidden layer
        hidden_layer = layers.Dense(100, activation="relu")(input_layer)

        # create output layer
        output_layer = layers.Dense(1, activation="softmax")(hidden_layer)

        classifier = models.Model(inputs=input_layer, outputs=output_layer)
        classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',
                           metrics=['accuracy', f1_m, tf.keras.metrics.Precision(),
                                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                           )
        return classifier

    #classifier = create_model_architecture(train_features_updated.shape[1])
    #accuracy_NN = train_model(classifier, train_features_updated, train_y, test_features_updated, valid_y, is_neural_net=True)
    #print("NN, Ngram Level TF IDF Vectors", accuracy_NN)
    #exit()

    #tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(trainDF['text'])
    word_index = tokenizer.word_index


    #exit()
    def make_matrix(embeddings_index):
        print('Making matrix...')
        all_embs = numpy.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        nb_words = min(MAX_NUM_WORDS, len(word_index))
        # embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def load_fasttext():
        print('Loading wiki-news embeddings...')
        EMBEDDING_FILE = 'Data/wiki-news-300d-1M.vec'

        def get_coefs(word, *arr):
            return word, numpy.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))
        return make_matrix(embeddings_index)

    print('Loading Wordvec...')
    embedding_matrix = load_fasttext()
    #exit()
    # for i, line in enumerate(open('Data/wiki-news-300d-1M.vec', 'r')):
    #    values = line.split()
    #    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    train_seq_x = sequence.pad_sequences(tokenizer.texts_to_sequences(train_x), padding='post', maxlen=500)
    valid_seq_x = sequence.pad_sequences(tokenizer.texts_to_sequences(valid_x), padding='post', maxlen=500)
    print(train_seq_x.shape)
    # CNN
    def create_cnn(input_size):
        # Add an Input Layer
        input_layer = layers.Input((input_size,), sparse=True)

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(128, 5, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(100, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.1)(output_layer1)
        output_layer2 = layers.Dense(1, activation="softmax")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Nadam(), loss='binary_crossentropy',
                      metrics=['accuracy', f1_m, tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

        return model

    classifier = create_cnn(train_seq_x.shape[1])
    accuracy_CNN_0 = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)
    print("CNN, Word Embeddings", accuracy_CNN_0)
    #exit()
# Recurrent Neural Network – LSTM

    def create_rnn_lstm(input_size):
        # Add an Input Layer
        input_layer = layers.Input((input_size,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the LSTM Layer
        lstm_layer = layers.LSTM(100)(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="softmax")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',
                      metrics=['accuracy', f1_m, tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                      )

        return model

    classifier = create_rnn_lstm(train_seq_x.shape[1])
    accuracy_RNN_LSTM = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)
    print("RNN-LSTM, Word Embeddings", accuracy_RNN_LSTM)

#Recurrent Neural Network – GRU
    def create_rnn_gru(input_size):
        # Add an Input Layer
        input_layer = layers.Input((input_size,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the GRU Layer
        lstm_layer = layers.GRU(100)(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="softmax")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',
                      metrics=['accuracy', f1_m, tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                      )

        return model

    classifier = create_rnn_gru(train_seq_x.shape[1])
    accuracy_RNN_GRU = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)
    print("RNN-GRU, Word Embeddings", accuracy_RNN_GRU)
    pyplot.show()
    exit()
# Bidirectional RNN

    def create_bidirectional_rnn():
        # Add an Input Layer
        input_layer = layers.Input((500,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the LSTM Layer
        lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="softmax")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',
                      metrics=['accuracy', f1_m, tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                      )

        return model

    classifier = create_bidirectional_rnn()
    accuracy_RNN_BiD = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)
    print("RNN-Bidirectional, Word Embeddings", accuracy_RNN_BiD)

# Recurrent Convolutional Neural Network
    def create_rcnn():
        # Add an Input Layer
        input_layer = layers.Input((500,))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the recurrent layer
        rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)

        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="softmax")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy',
                      metrics=['accuracy', f1_m, tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                      )
        return model
    classifier = create_rcnn()
    accuracy_CNN = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)
    print("CNN, Word Embeddings", accuracy_CNN)

# load excel with its path
wb = xlrd.open_workbook("C:/Users/sairo/Documents/ark/Project/DE/Posts_csv/Survior Corps_Posts/processed_batch2.xls")
sh = wb.sheet_by_name("1_1 - 1_2 Survivor Corp (1)")
texts= []
labels = []
for i in range(sh.nrows):
    texts.append(sh.cell_value(i, 0))
    labels.append(sh.cell_value(i, 1))
print("Start")
# word2vec.printInputData(texts, labels)
printInputData(texts, labels)




