from flask import Flask, request, render_template
import pickle
import numpy as np

# Load mô hình đã lưu
model = pickle.load(open('model.pkl', 'rb'))

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Trang chính
@app.route('/')
def home():
    return render_template('index.html')  # Giao diện người dùng

# Xử lý khi người dùng submit dữ liệu
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu người dùng nhập
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        # Dự đoán với mô hình
        prediction = model.predict(features_array)[0]

        # Trả kết quả ra giao diện
        return render_template('index.html', prediction_text=f'Kết quả phân loại: {prediction}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Lỗi: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
