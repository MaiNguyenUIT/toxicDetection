# Sử dụng Python 3.10 làm base image
FROM python:3.10

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy tất cả file từ thư mục hiện tại vào container
COPY . .

# Cài đặt thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose cổng 5000
EXPOSE 5000

# Chạy ứng dụng bằng Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "source.api:app"]
