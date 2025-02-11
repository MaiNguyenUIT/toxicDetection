import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from preprocess import preprocess_data

# Load dữ liệu
# Load dữ liệu
X_train, y_train, tokenizer = preprocess_data('D:/toxicDetection/data/youtoxic_english_1000.csv', 
                                              text_column='Text', 
                                              label_columns=['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist'])

# Xây dựng mô hình
EMBEDDING_DIM = 100
model = Sequential([
    Embedding(input_dim=50000, output_dim=EMBEDDING_DIM, input_length=250),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')  # Một đầu ra duy nhất
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Lưu mô hình và tokenizer
model.save('../models/lstm_toxic_model_binary.h5')

import pickle
with open('../models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

