from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from preprocess import clean_text, MAX_SEQUENCE_LENGTH

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình và tokenizer
model = tf.keras.models.load_model('../models/lstm_toxic_model_binary.h5')
with open('../models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Hàm xử lý dự đoán
def predict_toxicity(review, threshold=0.5):
    review_clean = clean_text(review)
    sequence = tokenizer.texts_to_sequences([review_clean])
    review_pad = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(review_pad)[0][0]
    return "Toxic" if prediction > threshold else "Not Toxic"

# Định nghĩa endpoint API
@app.route('/predict', methods=['POST', 'HEAD'])
def predict():
    try:
        if request.method == "HEAD":
            return "", 200  # HEAD requests return headers only, no body
        # Lấy dữ liệu từ yêu cầu
        data = request.json
        review = data.get('review', None)

        if not review:
            return jsonify({"error": "Review is required"}), 400

        # Gọi hàm dự đoán
        result = predict_toxicity(review)
        return jsonify({"Toxic": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
