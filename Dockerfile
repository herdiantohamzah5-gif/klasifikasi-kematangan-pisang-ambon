# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Set folder kerja di dalam kontainer
WORKDIR /app

# Salin file kebutuhan pustaka
COPY requirements.txt .

# Install pustaka Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kodingan proyek ke dalam kontainer
COPY . .

# Expose port bawaan Streamlit
EXPOSE 8501

# Jalankan aplikasi Streamlit saat kontainer di-start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]