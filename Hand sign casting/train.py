import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

try:
	data = pd.read_csv('hand_data.csv', header = None)

except FileNotFoundError:
	print("Error: The data file not found.")
	exit()

x = data.iloc[:, 1:]
y = data.iloc[:,0]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y_encoded,test_size=0.2, random_state= 42)

model = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, activation='relu')

print(f"Training on classes: {list(encoder.classes_)}")
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

with open('spell_model.pkl', 'wb') as f:
	pickle.dump((model, encoder), f)

print("Done! All set for the next move!")