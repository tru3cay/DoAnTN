from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Tải mô hình
model = joblib.load('dudoan_sanpham.pkl')

# Khởi tạo đối tượng LabelEncoder cho 'product_code'
le_product = LabelEncoder()

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ yêu cầu POST
    data = request.get_json(force=True)

    for key in data:
        if not isinstance(data[key], list):
            data[key] = [data[key]]
    df = pd.DataFrame(data)

    print(data)

    # Chuyển đổi và trích xuất các đặc trưng từ 'order_date'
    df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
    df['day'] = df['order_date'].dt.day
    df['month'] = df['order_date'].dt.month
    df['year'] = df['order_date'].dt.year


    # Mã hóa cột 'product_code'
    df['product_code_encoded'] = le_product.fit_transform(df['product_code'])

    # Chuẩn bị dữ liệu để dự đoán
    features = ['product_code_encoded', 'day', 'month', 'year', 'sales', 'cost_price', 'revenue_inr']
    X_real = df[features]

    # Dự đoán
    predictions = int(model.predict(X_real))


    # Trả về kết quả dự đoán
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
