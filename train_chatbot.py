import json
import pickle
import random

import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (only needs to be done once)
nltk.download('punkt')
nltk.download('wordnet')

# --- 1. Load and Preprocess Data ---

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Loop through each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Add the words to the words list
        # Associate words with their corresponding tag
        documents.append((word_list, intent['tag']))
        # Add the tag to the classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words, convert to lowercase, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save the processed words and classes to files for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# --- 2. Create Training Data (Bag-of-Words) ---

training = []
output_empty = [0] * len(classes)

# Create bag-of-words for each sentence
for doc in documents:
    bag = []
    word_patterns = doc[0]
    # Lemmatize and lowercase the pattern words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create the bag-of-words vector
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create the output row: 1 for the current tag, 0 for others
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data and convert to a NumPy array
random.shuffle(training)
training = np.array(training, dtype=object) # Use dtype=object to handle lists of different lengths

# Split the data into features (train_x) and labels (train_y)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


# --- 3. Build and Train the Neural Network ---

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
# Using Stochastic Gradient Descent (SGD) with Nesterov momentum
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
# epochs defines the number of times the model sees the entire dataset
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5', history)

print("Done! The model is trained and saved as chatbot_model.h5")