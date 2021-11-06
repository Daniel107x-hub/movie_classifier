from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequence(sequence, dimension=10000):
    results = np.zeros((len(sequence),dimension)) # Tensor del tamano (reviews, palabras)
    for i,sequence in enumerate(sequence):
        results[i,sequence] = 1. # Recolecta informacion de las palabras contenidas en cada resena, si una palabra aparece le asigna un 1
    return results

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index() # Extrae diccionario de Palabras->Numeros

reversed_word_index = dict(
    [(value,key) for (key,value) in word_index.items()]  # Mapea de Numeros->Palabras revirtiendo el diccionario
)
decoded_review = ' '.join(
    [reversed_word_index.get(i-3,'?') for i in train_data[0]] # Obtiene la palabra asociada a cada numero en el dato de entrenamiento 0, agregando un offset de 3 porque los numero 0, 1 y 2 se utilizan para otra cosa.
)

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

#Vectorizacion de etiquetas
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#Model
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 4,
                    batch_size=512,
                    validation_data=(x_val,y_val)
                    )

history_dict = history.history
history_dict.keys()

#View data
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

results = model.evaluate(x_test,y_test)
print(results)