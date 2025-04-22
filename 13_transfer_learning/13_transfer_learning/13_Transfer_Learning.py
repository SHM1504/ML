#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:44:14 2023

@author: d
"""

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

"""
=============================================================================
Transfer Learning
=============================================================================

Quelle:
    https://neptune.ai/blog/transfer-learning-guide-examples-for-images-and-text-in-keras


1. Was ist Transfer Learning:

- Es geht um die Nutzung von Merkmalsrepräsentationen eines bereits 
  trainierten Modells
- man muss ein neues Modell nicht von Grund auf trainieren 
- vortrainierte Modelle werden in der Regel anhand umfangreicher Datensätze 
  trainiert, die einen Standard-Benchmark in der Computer-Vision-Branche 
  darstellen. 
- daraus können die Gewichte in anderen Bildverarbeitungsaufgaben 
  wiederverwendet werden. 
- Modelle können direkt für Vorhersagen bei neuen Aufgaben verwendet 
- oder in den Prozess der Ausbildung eines neuen Modells integriert werden
- führt zu einer geringeren Trainingszeit und einem geringeren 
  Generalisierungsfehler.  
- besonders nützlich, wenn nur ein kleiner Trainingsdatensatz vorhanden. 
- kann auch auf Probleme der natürlichen Sprachverarbeitung angewendet werden 

"""

im = img.imread('13_Transfer_Learning_01-Überblick.png')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()

"""
Vorteil von Pretained Models: Sind allgemein genug,dass sie dann auf andere
real-world Probleme angewandt werden können, z.B.
    Modelle, die auf ImageNet trainert wurden (1000 Kategorien), können 
    z.B. auf Insekten spezialisiert werden.
    Text Klassifizierung erfordert die Kenntnis von Wortdarstellungen -> viele
    Trainingsdaten notwendig und langwierig
    
    
Unterschied zwischen transfer learning und fine-tuning

Fine-Tuning 
    - ist ein optionaler Prozess bei dem die Performance verbessert werden soll
    - ganzes Model muss immer neu trainiert werden
    - Gefahr des Overfitting
    - Overfitting kann vermieden werden durch
        - das Netz oder nur einen Teil mit niedrigerer Lernrate trainieren 
          (verhindert bedeutsame Updates, die eine schlechte Performance 
           hervorrufen können) oder einen early_stopping einbauen
          

Transfer Learning
    - Annahme: Man hat 100 Katzen- und 100 Hundebilder und man möchte diese 
      klassifizieren -> wenige Daten -> starkes Overfitting
    - Training auf hohe Genauigkeit benötigt viele Trainingsdaten!
    - ImageNet über 1 Million -> Dauert auch sehr lange! Hardware!
    
    
Wann funktioniert Transfer Learning nicht?
    - Wenn high-level Features, die die unteren Schichten gelernt haben, nicht 
      ausreichen, z.B. hat das Netz gelernt, Türen zu erkennen, aber nicht, ob
      diese offen oder geschlossen sind.
    - in einem solchen Fall nimmt man die low-level Features des vortrainierten
      Netzwerks und trainiert dann die spezielleren nach, oder nimmt nur die 
      Gewichte der ersten paar Layer
    - Wenn Datensätze nicht ähnlich sind, aber auch dann wird ein 
      vortrainiertes Netz besser performen als ein zufällig initialisiertes
    - Wenn man die Struktur des Netzes im Verhältnis zum antrainierten ändert,
      indem man z.B. Schichten entfernt -> kann dann sehr aufwändig werden!
    
      
=============================================================================
Wie implementiert man transfer learning? 6 Schritte:
=============================================================================

    
1. Das vortrainierte Model holen, z.B. hier: 
   https://keras.io/api/applications/

"""
#%%
im = img.imread('13_Transfer_Learning_02-Base.png')
plt.figure(dpi=600)
# plt.figure.figsize(16,10)
plt.axis("off")
plt.imshow(im)
plt.show()

"""
2.  Basis Model erstellen
    Man erstellt eine Basismodel nach einer bestimmten Architektur, z.B. ResNet 
    oder Xception und kann auch die Gewichte herunterladen.
    Die Output Layer wird wahrscheinlich nicht passen, also wird man diese
    erst mal entfernen müssen. Später wird dann eine passende eingesetzt.

3. Layers einfrieren ("freeze"), damit sie sich während des Trainings nicht ändern
   Wichtiger Schritt, weil man die Gewichte nicht verlieren will!
   
   base_model.trainable = False

4. Neue trainierbare Schichten am Ende einbauen
   Diese neuen Schichten machen aus den alten Features die Vorhersagen für das
   neue Data Set (wichtig, da ja die Outputlayer gelöscht wurde!).
   
5. Neue Schichten trainieren
   Sollte klar sein.

6. Fine-Tuning
   Optional
   Alle Schichten werden geöffnet ("unfreeze") und trainiert
   LR muss klein sein, da das Modell groß ist und der Datensatz klein
   Dann rekompilieren, weil dann der Zustand wieder eingefroren wird
   Immer wenn man das Verhalten ändern will, muss man neu kompilieren.



Wo bekommt man vortrainierte Modelle? 

1. keras.applications

https://keras.io/api/applications/
Wenn man ein Model herunterlädt, sind die Gewichte automatisch mit dabei.
Gespeichert in "~/.keras/models/."
Alle keras.applications werden für Bilder genutzt.

Beispiel:
    model = tf.keras.applications.MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
)



2. Tensorflow

https://www.tensorflow.org/hub


Beispiel:
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
                       trainable=False),
            tf.keras.layers.Dense(num_classes, activation='softmax')
            ])


3. Word Embedding
Für Text Klassizierung:
    - https://nlp.stanford.edu/projects/glove/
    - https://code.google.com/archive/p/word2vec/   trainiert an 1 Billion Wörtern von Google News
    - https://fasttext.cc/docs/en/english-vectors.html  für englische Wörter
    
weiterführende Infos:   
    https://neptune.ai/blog/word-embeddings-deep-dive-into-custom-datasets
    

4. Hugging Face
    - ebenfalls für Aufgaben mit Texten wie
        - Fragen beantworten
        - Zusammenfassen
        - Übersetzung
        - Text Generierung
        - ...
    - über 100 Sprachen werden unterstützt
    
Beispiel:
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis')
    classifier('We are very happy to include pipeline into the transformers repository.')

    Ausgabe: [{'label': 'POSITIVE', 'score': 0.9978193640708923}]

"""
#%%
"""   
=============================================================================
Wo benutzt man pretrained models?
=============================================================================

1. Vorhersage

Model herunterladen und sofort Vorhersagen machen, z.B. mit
https://keras.io/api/applications/resnet/#resnet50-function
ist auf ImageNet trainiert

"""

model = ResNet50(weights='imagenet')
img_path = '13_Transfer_Learning_03-elephant2.jpg'
imag = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(imag)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

im = img.imread('13_Transfer_Learning_03-elephant.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()


im = img.imread('13_Transfer_Learning_03-elephant2.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()



#%%
preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

# Elephant - nicht quadratisch
# Predicted: [('n02504458', 'African_elephant', 0.45957744), ('n01871265', 'tusker', 0.43963525), ('n02504013', 'Indian_elephant', 0.0994692)]

# Elephant2 - quadratisch
# Predicted: [('n02504458', 'African_elephant', 0.6601476), ('n01871265', 'tusker', 0.2198429), ('n02504013', 'Indian_elephant', 0.111071594)]


#%% 

"""
=============================================================================
2. Feature Extraction
=============================================================================

Der Output der vorletzten Layer wird in ein neues Model als Input gegeben.
Ziel: das bereits trainierte Modell oder einen Teil davon wird verwendet, um 
      Bilder vorzuverarbeiten und wesentliche Merkmale zu erhalten. 
      Anschließend neuer Classifier, das Basismodell muss nicht neu trainiert 
      werdem

Beispiel: word embedding wird für feature extraction benutzt, Wörter werden in
    Zusammenhang und die korrekte Position im Vektor gebracht.
    Also sind sie unabhängig von der eigentlichen Aufgabe im NLP.
    Englische Modelle können auch mit deutscher Sprache verwendet werden.
    
"""
im = img.imread('13_Transfer_Learning_04-Feature_extraction.jpg')
plt.figure(dpi=600)
plt.axis("off")
plt.imshow(im)
plt.show()


"""
=============================================================================
3. Fine-Tuning
=============================================================================

Wenn der neue Classifier fertig ist, kann er noch getuned werden, dazu muss der 
Classifier oder ein Teil davon "entfroren" ("aufgetaut") werden. Ist notwendig, 
wenn die Feature Represenations für das eigentliche Problem relevanter gemacht
werden sollen.

Man kann dazu das Model auch mit den antrainierten Gewichten initialieren - 
das hängt alles vom Problem ab.
"""




#%%
"""
=============================================================================
Beispiel für Bilder in Keras
=============================================================================
"""

# Datei herunterladen
# url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# filename = 'catsdogs.zip'
# urllib.request.urlretrieve(url, filename)

# Verzeichnis erstellen und ZIP-Datei extrahieren
path = os.getcwd()
base_dir = os.path.join(path, "cats_and_dogs_filtered")
# base_dir = os.path.normpath('/content/cats_and_dogs_filtered')  # Passe den Pfad zu dem gewünschten Verzeichnis an
# os.makedirs(base_dir, exist_ok=True)
# with zipfile.ZipFile(filename, 'r') as zip_ref:
#     zip_ref.extractall(base_dir)

# Trainings- und Validierungsverzeichnisse festlegen
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

#%% Erstellen der Train und Validation Sets

training_set = image_dataset_from_directory(train_dir,
                                            shuffle=True,
                                            batch_size=32,
                                            image_size=(150, 150))

val_dataset = image_dataset_from_directory(validation_dir,
                                           shuffle=True,
                                           batch_size=32,
                                           image_size=(150, 150))


#%% Augmentation - verringert Overfitting 

# Augmentation Model

# data_augmentation = keras.Sequential(
#     [keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
#       keras.layers.experimental.preprocessing.RandomRotation(0.1),]
#     )

data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip("horizontal"),
     keras.layers.RandomRotation(0.1),]
    )

#%% 

for images, labels in training_set.take(1):
    plt.figure(figsize=(12, 12))
    first_image = images[0]
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        augmented_image = data_augmentation(
            tf.expand_dims(first_image, 0),training=True   # TRAINING WICHTIG!!!
        )
        plt.imshow(augmented_image[0].numpy().astype("int32"))
        plt.axis("off")
    plt.show()

#%% Creat a base model

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)   # letzte Schicht wird nicht mit importiert

base_model.trainable = False  # Freeze!

#%% Final Dense Layer

# Vorher noch den Input:

inputs = keras.Input(shape=(150, 150, 3))

x = data_augmentation(inputs) 

x = tf.keras.applications.xception.preprocess_input(x)

x = base_model(x, training=False)  # Batch Normalizazion Layers bekommen keine Updates
x = keras.layers.GlobalAveragePooling2D()(x)  # convert features from the base model to vectors
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1)(x)  # final dense layer
model = keras.Model(inputs, outputs)


#%% Train the model

model.compile(optimizer='adam', 
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=keras.metrics.BinaryAccuracy())

model.fit(training_set, 
          epochs=20, 
          validation_data=val_dataset)


#%% Fine Tuning -> Unfreeze

base_model.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=keras.metrics.BinaryAccuracy())

#%% EarlyStopping

log_path = "Logs2"
if not os.path.exists(log_path):
    os.mkdir(log_path)

# in anaconda prompt: tensorboard --logdir PATH_TO_LOG_FOLDER
log_folder = 'Logs2'
callbacks = [
            EarlyStopping(patience = 5),
            TensorBoard(log_dir=log_folder)
            ]

#%% Fit

model.fit(training_set, epochs=15, validation_data=val_dataset, callbacks=callbacks)

#%% Beispiel mit NLP

"""
=============================================================================
Example of transfer learning with natural language processing
=============================================================================

Word Embedding ist ein Vector, der einen Text repräsentiert, Wörter mit 
ähnlicher Bedeutung sind näher

"""

#%% Datensatz laden   

df = pd.read_csv('combined_data.csv')

print(df.head())

#%% Aufteilen

X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#%% Vorbereiten für die Verarbeitung

vocab_size = 10000
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)  
# Verwandelt Sätze in numerische Repräsentationen
## oov: Out Of Vocabulary, sammelt alle Wörter, die nicht im Vocabulary drin sind
tokenizer.fit_on_texts(X_train)

# Word index um zu sehen, wie aus Wörtern Nummern werden
word_index = tokenizer.word_index
print(word_index)

#%% Erstellen von Sequenzem - Sätze werden durch diese Sequenzen repräsentiert

X_train_sequences = tokenizer.texts_to_sequences(X_train)

# DIESE ZEILE FEHLT IN DER QUELLE
X_test_sequences = tokenizer.texts_to_sequences(X_test)

print(X_train_sequences[0:2])

#%% Als Input in ein Netz müssen alle Sequenzen gleich lang sein:

padding_type='post'  # Nullen am Ende von kurzen Sequenzen
truncation_type='post'  # Lange Sequenzen werden hinten abgeschnitten
max_length = 100

# DIESE ZEILEN FEHLEN IN DER QUELLE
X_train_padded = pad_sequences(X_train_sequences, padding=padding_type, 
                               truncating=truncation_type, maxlen=max_length)

X_test_padded = pad_sequences(X_test_sequences, padding=padding_type, 
                               truncating=truncation_type, maxlen=max_length)

#%% Transfer Learning

"""
GloVe (Global Vectors for Word Representation) sind vortrainierte
word embeddings

http://nlp.stanford.edu/data/glove.6B.zip

Ein Embedding-Vector ist eine numerische Repräsentation eines Wortes.
Mit Hilfe von solchen Vektoren werden statistische Zusammenhänge oder
Ähnlichkeiten zwischen Wörtern dargestellt.
Diese Vektoren wurden von NNs erstellt.

Beispiel: Vektoren von "Hund" und "Katze" sind sich ähnlicher als "Katze" und
"Auto"

"""

# Ein Dictionary aus der Embeddings-Datei aufbauen
embeddings_index = {}

f = open('/mnt/usb-Seagate_Expansion+_NAAG83MS-0:0-part2/Datenschatz/glove.6B/glove.6B.100d.txt')

for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    
f.close()

# Warten auf die 400 000

print('\nFound %s word vectors.' % len(embeddings_index))

#%%

"""
Embedding Matrix wird erstellt, sie dient als Lookup-Tabelle für die Vektoren.

Die Dimension der Embedding-Matrix ergibt sich aus der Anzahl der Wörter im 
Vokabular und der Dimension der Embedding-Vektoren.
"""

embedding_matrix = np.zeros((len(word_index) + 1, max_length))   
#  Eins mehr als len, weil 0 für OOV reserviert ist.
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embeddings_index.get("bakery")

#%% Embedding Layer wird erstellt:  
    
embedding_layer = Embedding(len(word_index) + 1,  # 0 für OOV
                            max_length,
                            weights=[embedding_matrix],  # Gewichte sind dadurch vortrainiert
                            input_length=max_length,
                            trainable=False)  # vortrainiert! Nicht anfassen!

#%% Model erstellen

"""
 Bidirectional LSTM: Ein Long-Short-Term-Memory-Model, das in beide Richtungen 
 funktioniert, besteht aus zwei unabhängigen Untereinheiten, eine vorwärts und
 eine rückwärts.
"""
model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(150, return_sequences=True)),
    Bidirectional(LSTM(150)),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

#%% Modell compilieren

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%  EarlyStopping

log_folder = 'Logs3'
callbacks = [
            EarlyStopping(patience = 10),
            TensorBoard(log_dir=log_folder)
            ]
num_epochs = 60

#%% fit ca. 50 Epochen reichen

history = model.fit(X_train_padded, y_train, epochs=num_epochs, validation_data=(X_test_padded, y_test), callbacks=callbacks)

#%% Performance anzeigen

loss, accuracy = model.evaluate(X_test_padded, y_test)
print('Test accuracy :', accuracy)

#%% Versuch einer Prediction

# df erstellen
satz = "Absolutely stunning! Great Job!"
satz1 = "The worst thing I ever used"
satz_series = pd.Series(satz)
satz1_series = pd.Series(satz1)

# tokenizer
satz_sequences = tokenizer.texts_to_sequences(satz_series)
satz1_sequences = tokenizer.texts_to_sequences(satz1_series)

# Padding
satz_padded = pad_sequences(satz_sequences, padding=padding_type, 
                            truncating= truncation_type, maxlen=max_length)
satz1_padded = pad_sequences(satz1_sequences, padding=padding_type, 
                             truncating= truncation_type, maxlen=max_length)

#%% Prediction

prediction = model.predict(satz_padded)

prediction1 = model.predict(satz1_padded)

print("Positives Feedback (sollte 1 sein)", prediction)
print("Negatives Feedback (sollte 0 sein)", prediction1)
