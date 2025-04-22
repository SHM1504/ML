# -*- coding: utf-8 -*-
"""

"""

# =============================================================================
# 06_ Autoencoder
# =============================================================================

# https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/

# Autoencoder: künstliches neuronales Netzwerk, das darauf spezialisiert ist,
#              Eingabedaten effizient zu komprimieren (kodieren), um sie auf ihre wesentlichen 
#              Merkmale (wichtigsten Informationen der ursprünglichen Eingabe) zu reduzieren, und 
#              dann die ursprüngliche Eingabe aus dieser komprimierten Darstellung zu rekonstruieren (dekodieren)
#              (Bilder, Texte, Zeitreihen, Genetische Daten, Audiodateien)

# unüberwachtes maschinelles Lernen 

# Dimensionalitätsreduktion und Feature Extraction

# Dabei geht ein Teil der Informationen verloren, da die 
# Komprimierung verlustbehaftet und datenspezifisch ist

# Datenspezifisch: 
# Autoencoder kann nur die Daten effektiv komprimieren mit denen er trainiert wurde. 
# -> ein mit Hunde-Bildern trainierter Autoencoder würde schlechte Ergebnisse 
# für Katzen-Bilder liefern

# Verlustbehaftete Operationen: 
# Das rekonstruierte Bild oft nicht so scharf oder hochauflösend wie das Original 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


img = mpimg.imread("05_Autoencoder/06_Autoencoder/Encoder_Decoder.png")
plt.figure(figsize =(16,9), dpi=150)
plt.axis("off")
plt.title("Encoder - Decoder", fontsize = 32, loc="center", x=.5, y=.9)
plt.imshow(img)
plt.show()



# 1: Einfacher Autoencoder
# 2: Tiefgreifender CNN-Autoencoder
# 3: Rauschunterdrückung des Autoencoders

# %%


from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D   # Dense: voll verbundene (fully connected) Schichten 
                                                                  # Conv2D: 2D-Convolutional Layer (Verarbeitung von 2D-Bildern)
                                                                  # MaxPooling2D: - räumliche Dimension eines Bildes reduzieren 
                                                                  #               - wichtigstes Merkmal auss einem Bereich extrahieren
                                                                  # UpSampling2D: räumliche Auflösung eines Bildes erhöhen 

                                                                                                                               
from keras import Input, Model             # Input: Eingabeform des Modells definieren
                                           # Model: Modell erstellen
                                          
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential



# Information: 
# TensorFlow verwendet eine Optimierung für die Berechnungen auf der CPU
# -> Unterschiedliche Ergebnisse durch Rundungsfehler möglich (normalerweise unbedenklich)
    
    
# %%
# =============================================================================
# 1. Einfaches Autoencoder-Modell mit Keras erstellen 
# =============================================================================

# =============================================================================
# 1.1. Definieren der Variablen und Eingabeform:   
# =============================================================================

encoding_dim = 15                   # encoding_dim = 15: Dimension des kodierten Vektors  
                                    # -> Anzahl Neuronen in der komprimierten Darstellung des Eingabebildes 
                                    
                                    # Anzahl der Dimensionen: wie stark wird die Eingabe komprimiert
                                    # -> je kleiner die Dimension, desto größer ist die Kompression
                                    
input_img = Input(shape=(784,))     # Eingabeknoten des Modells (Vektor mit 784 Werten; Bild mit 28x28 Pixeln)


# %%
# =============================================================================
# 1.2. Erstellen des encodierten (komprimierten) Teils
# =============================================================================

# => Eingabedaten von 784 Dimensionen (28×28 Pixel) auf 15 Dimensionen komprimiert
# => zwingt das Modell, die wichtigsten Merkmale der Bilder zu extrahieren

encoded = Dense(encoding_dim, activation='relu')(input_img)  # encoded: encodierte (komprimierte) Teil des Autoencoders
                                                             # Dense-Schicht: Ausgabegröße von encoding_dim (also 15)
                                                             # ReLU-Aktivierungsfunktion (nicht-lineare Beziehungen)
                                                             # Eingabe: input_img-Tensor  (der das Bild darstellt)
# encoded: speichert latenten Vektor
# - komprimierte, reduzierte Darstellung der Daten
# - fasst die wichtigsten Merkmale zusammen
# - kann für weitere Analysen, Visualisierungen oder als Input 
#   für andere Machine-Learning-Modelle verwendet werden

# %%
# =============================================================================
# 1.3. Erstellen des decodierten Teils 
# =============================================================================

decoded = Dense(784, activation='sigmoid')(encoded)  # decoded: decodierter Teil des Autoencoders
                                                     # Dense-Schicht: Daten zurück in die ursprüngliche Form 
                                                     #                (784 Werte) rekonstruieren
                                                     # Sigmoid-Aktivierungsfunktion: Ausgangswert des decodierten 
                                                     # Bildes soll zwischen 0 und 1 liegen -> typisch für Bilddaten


# %%
# =============================================================================
# 1.4. Erstellen des gesamten Autoencoders
# =============================================================================

autoencoder = Model(input_img, decoded)   # Input (input_img) und Output (decoded) werden verbunden
                                          

# Information: TensorFlow greift auf die leistungsfähigsten verfügbaren Instruktionssätze zu



# %%
# =============================================================================
# 1.5. Erstellen separater Encoder-und Decoder 
# =============================================================================

# - Encoder- und Decoder-Teile des Autoencoders können später unabhängig voneinander verwendet werden
# - erleichtert den Zugriff auf die komprimierten Repräsentationen und rekonstruierten Daten


# =============================================================================
# Erstellen des Encoder-Modells
# =============================================================================

encoder = Model(input_img, encoded)  # Erstellen des Encoder-Modells
                                     # Input: Das Eingabebild (input_img) (Dimension von (784,))
                                     # Output: Die kodierte Repräsentation (encoded) (Dimension 15)



encoded_input = Input(shape=(encoding_dim,))  # Eingang für den Decoder, damit er weiß, dass er mit 15 Werten beginnt



# =============================================================================
# Decoder extrahieren
# =============================================================================

decoder_layer = autoencoder.layers[-1]   # letzte Schicht des ursprünglichen Autoencoders extrahieren
                                         # Schicht, die die Rekonstruktion des Bildes durchführt (784 Ausgänge)


decoder = Model(encoded_input, decoder_layer(encoded_input)) # Erstellen des Decoder-Modells
                                                             # Input: komprimierte Repräsentation (encoded_input)
                                                             # Output: Rekonstruktion des Bildes (mit 784 Dimensionen)


# %%
# =============================================================================
# 1.6. Modell kompilieren (ADAM-Optimierer und Kreuzentropieverlustfunktion)
# =============================================================================

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# %%
# =============================================================================
# 1.7. Daten laden
# =============================================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Laden des MNIST-Datensatzes


x_train = x_train.astype('float32') / 255.   # Normalisieren der Werte
x_test = x_test.astype('float32') / 255.     # Pixelwerte: zwischen 0 und 255 (Grauwerte)
                                             # Division durch 255: Werte auf 0 bis 1 skalieren
                                             

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))   # Umformen der Bilder in flache Vektoren (2D->1D)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))       # -> 28 × 28 = 784 (Gesamtzahl der Pixel)
                                                                        # Format vor der Umwandlung:
                                                                        # x_train.shape → (60000, 28, 28)
                                                                        # x_test.shape → (10000, 28, 28)
print(x_train.shape)
print(x_test.shape)

# (60000, 784)    # 60.000 Trainingsbilder, jedes Bild wird als ein Vektor mit 784 Werten dargestellt
# (10000, 784)

# %%
# =============================================================================
# 1.8. Visualisierung eines Bildes
# =============================================================================

plt.imshow(x_train[0].reshape(28,28))   # x_train[0]: erste Bild, aber in Vektorform (784,)
                                        # reshape(28,28) bringt es zurück in die ursprüngliche Form


# %%
# =============================================================================
# 1.9. Modell trainieren:
# =============================================================================

autoencoder.fit(x_train, x_train,           # Zielwerte identisch mit den Eingabedaten:
                epochs=15,                  # => Ziel des Trainings: die gleichen Daten zu rekonstruieren
                batch_size=256,
                validation_data=(x_test, x_test)) # Modell wird mit den Testdaten (x_test) nach jeder Epoche überprüft

    

# %%
# =============================================================================
# 1.10. Modell auf den Testdaten anwenden
# =============================================================================


encoded_img = encoder.predict(x_test)       # Kodierung der Bilder 
                                            # Encoder wird auf die Testdaten (x_test) angewendet



decoded_img = decoder.predict(encoded_img)  # Dekodierung der kodierten Bilder (Verwendung des Decoder-Modells)
                                            # Decoder wird auf die kodierten Repräsentationen angewendet



# Visualisierung der Ergebnisse: 
# Originalbilder aus dem Testdatensatz und deren rekonstruierte Versionen werden nebeneinander dargestellt
    

plt.figure(figsize=(20, 4))
for i in range(5):
    ax = plt.subplot(2, 5, i + 1)                # Originalbild anzeigen
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    ax = plt.subplot(2, 5, i + 1 + 5)            # Rekonstruiertes Bild anzeigen
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 
plt.show()


# %%
# =============================================================================
# 2. Tiefer CNN-Autoencoder
# =============================================================================

# Bei Bildern ist es sinnvoller, ein Convolutional Neural Network (CNN) zu verwenden: 
# Encoder: besteht aus einem Stack aus Conv2D- und Max-Pooling-Layer 
# Decoder: besteht aus einem Stack aus Conv2D- und Upsampling-Layer

# =============================================================================
# 2.1. Modell erstellen
# =============================================================================

model = Sequential()      # sequentielles Modell definieren


# =============================================================================
# Encoder: 
# =============================================================================

model.add(Conv2D(30, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))   
# 2D-Faltungsschicht (Conv2D): 
# 30 Filter/Kernel: um Merkmale im Eingabebild zu extrahieren
# 3: Größe des Kernels ist 3x3
# ReLU-Aktivierungsfunktion (nichtlineare Muster)
# padding='same': Ausgangsgröße = Eingabedimension
# input_shape=(28,28,1): Eingabebilder sind 28x28 Pixel groß und haben 1 Kanal (Graustufenbilder)


model.add(MaxPooling2D(2, padding= 'same'))
# Pooling-Schicht: Pooling-Fenster 2x2 (Bildgröße reduzieren) 
# padding='same': Ausgangsgröße=Eingabedimension


model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
# Faltungsschicht: 15 Kernel (3x3): weitere Merkmale extrahieren, Dimension weiter komprimieren
# padding='same': Ausgangsgröße = Eingabedimension


model.add(MaxPooling2D(2, padding= 'same'))
# Pooling-Schicht: Pooling-Fenster 2x2 (Bildgröße reduzieren) 
# padding='same': Ausgangsgröße = Eingabedimension



# =============================================================================
# Decoder: 
# =============================================================================

model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
# Faltungsschicht (Conv2D): komprimierte Daten entschlüsseln und Merkmale extrahieren
# 15 Kernel; 3x3
# ReLU-Aktivierungsfunktion
# padding='same': Ausgangsgröße = Eingabedimension


model.add(UpSampling2D(2))
# UpSampling-Schicht: vergrößert das Bild um den Faktor 2 
# => Erhöht die Dimensionen des Bildes, indem es die Pixelwerte dupliziert, 
# um das Bild zu vergrößern


model.add(Conv2D(30, 3, activation= 'relu', padding='same'))
# Faltungsschicht: 30 Kernel, 3x3, Relu, padding='same': Ausgangsgröße = Eingabedimension


model.add(UpSampling2D(2))
# UpSampling-Schicht: Vergößerung Faktor 2


model.add(Conv2D(1,3,activation='sigmoid', padding= 'same')) # output layer
# letzte Faltungsschicht: 1 Kernel, gibt die rekonstruierten Bildwerte zurück
# Aktivierungsfunktion: sigmoid (Werte zwischen 0 und 1)
# padding='same': Ausgangsgröße = Eingabedimension



# =============================================================================
# Komplilierung des Modells + Zusammenfassung
# =============================================================================

model.compile(optimizer= 'adam', loss = 'binary_crossentropy')

model.summary()



# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d (Conv2D)                 │ (None, 28, 28, 30)     │           300 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d (MaxPooling2D)    │ (None, 14, 14, 30)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_1 (Conv2D)               │ (None, 14, 14, 15)     │         4,065 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ max_pooling2d_1 (MaxPooling2D)  │ (None, 7, 7, 15)       │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_2 (Conv2D)               │ (None, 7, 7, 15)       │         2,040 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ up_sampling2d (UpSampling2D)    │ (None, 14, 14, 15)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_3 (Conv2D)               │ (None, 14, 14, 30)     │         4,080 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ up_sampling2d_1 (UpSampling2D)  │ (None, 28, 28, 30)     │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ conv2d_4 (Conv2D)               │ (None, 28, 28, 1)      │           271 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 10,756 (42.02 KB)
#  Trainable params: 10,756 (42.02 KB)
#  Non-trainable params: 0 (0.00 B)


# %%
# =============================================================================
# 2.1. Daten laden und das Modell trainieren
# =============================================================================

(x_train, _), (x_test, _) = mnist.load_data()    # Laden des MNIST-Datensatzes

x_train = x_train.astype('float32') / 255.       # Normalisieren der Bilddaten
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # Reshapen der Daten von (Anzahl, 28, 28) 
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))     # zu (Anzahl, 28, 28, 1): Kanaldimension hinzufügen
                                                          # 1 = Graustufenbilder

model.fit(x_train, x_train,          # Training des Autoencoders: x_train als Eingabe und Ziel
                epochs=15,
                batch_size=128,
                validation_data=(x_test, x_test)) # Modell wird mit den Testdaten (x_test) nach jeder Epoche überprüft


# %%
# =============================================================================
# 2.2. Visualisierung der Ergebnisse
# =============================================================================

pred = model.predict(x_test)       # erzeugt die rekonstruierten Versionen der 
                                   # Testbilder durch das Autoencoder-Modell

plt.figure(figsize=(20, 4))        

for i in range(5):
    
    ax = plt.subplot(2, 5, i + 1)            # Anzeigen der Originalbilder
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    ax = plt.subplot(2, 5, i + 1 + 5)        # Anzeigen der rekonstruierten Bilder
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()


# %%
# =============================================================================
# 3. Rauschunterdrückung des Autoencoders
# =============================================================================

# Rauschen: verschwommene Bilder
#           Veränderung der Farbe der Bilder
#           weiße Markierungen auf dem Bild


# =============================================================================
# 3.1. Rauschen simulieren
# =============================================================================

noise_factor = 0.7      # Rauschfaktor: legt fest, wie stark das Rauschen ist


# Hinzufügen von Rauschen zu den Trainings- und Testbildern:
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

# np.random.normal(loc=0.0, scale=1.0, size=x_train.shape):
# - erzeugt zufällige Werte aus einer Normalverteilung in der Form von x_train
# - zufällige Werte werden mit noise_factor (0.7) multipliziert (skaliert das Rauschen)
# - Rauschen zu den Originalbildern (x_train bzw. x_test) addieren
# => Die Bilder enthalten nun zusätzliche Störungen


# Begrenzen der Pixelwerte: 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)   # np.clip(): tellt sicher, dass alle Pixelwerte in den 
x_test_noisy = np.clip(x_test_noisy, 0., 1.)     # verrauschten Bildern im Bereich 0 bis 1 bleiben



# Visualisierung der verrauschten Testbilder:
plt.figure(figsize=(20, 2))
for i in range(1, 5 + 1):
    ax = plt.subplot(1, 5, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# %%
# ===============================================================================
# 3.2. Modells modifizieren (Filter/Kernel erhöhen), kompillieren und trainieren
# ===============================================================================


model = Sequential()

# Encoder
model.add(Conv2D(35, 3, activation= 'relu', padding='same', input_shape = (28,28,1))) # jetzt 35; vorher 30
model.add(MaxPooling2D(2, padding= 'same'))
model.add(Conv2D(25, 3, activation= 'relu', padding='same'))                          # jetzt 25; vorher 15
model.add(MaxPooling2D(2, padding= 'same'))

# Decoder
model.add(Conv2D(25, 3, activation= 'relu', padding='same'))                          # jetzt 25; vorher 15
model.add(UpSampling2D(2))
model.add(Conv2D(35, 3, activation= 'relu', padding='same'))                          # jetzt 35; vorher 30 
model.add(UpSampling2D(2))
model.add(Conv2D(1,3,activation='sigmoid', padding= 'same'))                          # output layer


model.compile(optimizer= 'adam', loss = 'binary_crossentropy')     # Modell kompiliieren

model.fit(x_train_noisy, x_train,                                  # Modell trainieren
                epochs=15,
                batch_size=128,
                validation_data=(x_test_noisy, x_test))




# %%
# =============================================================================
# 3.3. Visualisierung der Ergebnisse
# =============================================================================

pred = model.predict(x_test_noisy)

plt.figure(figsize=(20, 4))

for i in range(5):
    
    ax = plt.subplot(2, 5, i + 1)                   # Originalbilder anzeigen
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, 5, i + 1 + 5)               # Rekonstruktionen anzeigen
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()



















