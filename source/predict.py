import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text, MAX_SEQUENCE_LENGTH
import pickle

# Load mô hình và tokenizer
model = tf.keras.models.load_model('../models/lstm_toxic_model_binary.h5')

with open('../models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Hàm dự đoán toxic
def predict_toxicity(review, tokenizer, threshold=0.5):
    review_clean = clean_text(review)
    sequence = tokenizer.texts_to_sequences([review_clean])
    review_pad = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    
    prediction = model.predict(review_pad)[0][0]
    return "Toxic" if prediction > threshold else "Not Toxic"

# Chạy thử nghiệm
if __name__ == "__main__":
    review = input("Nhập review cần kiểm tra: ")
    result = predict_toxicity(review, tokenizer)
    print("Kết quả:", result)
