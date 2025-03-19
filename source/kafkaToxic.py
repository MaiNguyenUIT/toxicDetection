from confluent_kafka import Consumer, Producer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from preprocess import clean_text, MAX_SEQUENCE_LENGTH
import json

# Load mô hình và tokenizer
model = tf.keras.models.load_model('../models/lstm_toxic_model_binary.h5')
with open('../models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Cấu hình Kafka
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'toxicity-processor',
    'auto.offset.reset': 'earliest'
}
producer_conf = {'bootstrap.servers': 'localhost:9092'}

# Tạo Kafka Consumer và Producer
consumer = Consumer(consumer_conf)
producer = Producer(producer_conf)

# Đăng ký topic để đọc yêu cầu
consumer.subscribe(['toxicity-request'])

# Hàm xử lý dự đoán
def predict_toxicity(review, threshold=0.5):
    review_clean = clean_text(review)
    sequence = tokenizer.texts_to_sequences([review_clean])
    review_pad = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(review_pad)[0][0]
    return "Toxic" if prediction > threshold else "Not Toxic"

# Hàm xử lý tin nhắn từ Kafka
def process_message():
    while True:
        msg = consumer.poll(1.0)  # Chờ tối đa 1 giây
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        # Lấy request_id và review từ tin nhắn
        request_id = msg.key().decode('utf-8')
        review = msg.value().decode('utf-8')

        try:
            # Dự đoán tính độc hại
            result = predict_toxicity(review)
            print(f"Processed: {review} -> {result}")

            # Gửi kết quả đến topic toxicity-response
            producer.produce('toxicity-response', key=request_id.encode('utf-8'), value=result.encode('utf-8'))
            producer.flush()

        except Exception as e:
            print(f"Error processing message: {e}")
            # Có thể gửi lỗi về topic khác nếu cần
            producer.produce('toxicity-response', key=request_id.encode('utf-8'), value=f"Error: {str(e)}".encode('utf-8'))
            producer.flush()

# Chạy ứng dụng
if __name__ == '__main__':
    print("Starting Kafka consumer for toxicity prediction...")
    process_message()