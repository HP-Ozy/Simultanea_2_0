import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def pad_or_truncate(data, target_length):
    """Pad or truncate the data to the target length."""
    padded_data = []
    for item in data:
        if len(item) < target_length:
            # Pad with zeros
            padded_item = np.pad(item, (0, target_length - len(item)), 'constant')
        else:
            # Truncate to the target length
            padded_item = item[:target_length]
        padded_data.append(padded_item)
    return np.array(padded_data)

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Trova la lunghezza massima degli elementi
max_length = max(len(item) for item in data)

# Applica padding o truncation
data = pad_or_truncate(data, max_length)
labels = np.asarray(labels)

# Dividi i dati in set di addestramento e di test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Crea e addestra il modello Random Forest
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predice le etichette per i dati di test
y_predict = model.predict(x_test)

# Calcola la precisione del modello
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Salva il modello addestrato su un file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
