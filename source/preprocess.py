import pandas as pd
import numpy as np
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Tải stopwords
nltk.download('stopwords')
nltk.download('wordnet')  # Để dùng lemmatizer

from nltk.stem import WordNetLemmatizer

# Các thông số xử lý dữ liệu
MAX_NB_WORDS = 50000  # Số lượng từ tối đa
MAX_SEQUENCE_LENGTH = 250  # Độ dài tối đa của câu

# Tải lemmatizer
lemmatizer = WordNetLemmatizer()

# Hàm làm sạch văn bản
def clean_text(text):
    text = str(text).lower()  # Chuyển thành chữ thường
    text = re.sub(r'\d+', '', text)  # Loại bỏ số
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = re.sub(r'\s+', ' ', text).strip()  # Loại bỏ khoảng trắng thừa
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
    return text

# Hàm xử lý dữ liệu
def preprocess_data(file_path, text_column='Text', label_columns=['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist']):
    # Đọc file CSV
    df = pd.read_csv(file_path)

    # Kiểm tra dữ liệu null và làm sạch văn bản
    df = df.dropna(subset=[text_column] + label_columns)
    df['clean_text'] = df[text_column].apply(clean_text)

    # Gộp nhãn thành một nhãn `Toxic`
    df['Toxic'] = df[label_columns].any(axis=1).astype(int)  # 1 nếu bất kỳ nhãn nào là TRUE, ngược lại 0

    # Khởi tạo tokenizer
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df['clean_text'])

    # Chuyển đổi văn bản thành chuỗi số
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    data_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return data_pad, df['Toxic'].values, tokenizer

