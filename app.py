from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # รับค่าจาก form
    age = int(request.form['age'])
    unenconded_gender = request.form['gender']
    unenconded_education = request.form['education']
    unenconded_marital = request.form['marital']
    months = int(request.form['months'])
    credit = float(request.form['credit'])
    trans_amt = float(request.form['trans_amt'])
    trans_count = int(request.form['trans_count'])
    income = float(request.form['income'])
    
    # เข้ารหัสค่าสตริงให้เป็นตัวเลข
    gender = 1 if unenconded_gender == 'F' else 0
    education_mapping = {'High School': 0, 'Graduate': 1, 'Uneducated': 2, 'Unknown': 3, 'College': 4, 'Post-Graduate': 5, 'Doctorate': 6}
    education = education_mapping[unenconded_education]
    marital_mapping = {'Married': 0, 'Single': 1, 'Unknown': 2, 'Divorced': 3}
    marital = marital_mapping[unenconded_marital]
    
    # โหลดโมเดลจากไฟล์
    model = joblib.load('decision_tree_model.pkl')
    
    # สร้าง input data สำหรับ prediction พร้อมชื่อคุณลักษณะ
    input_data = {
        'Age': age,
        'Gender': gender,
        'Education_Level': education,
        'Marital_Status': marital,
        'Months_on_book': months,
        'Credit_Limit': credit,
        'Total_Trans_Amt': trans_amt,
        'Total_Trans_Count': trans_count,
        'Max_income': income,
    }
    
    input_data_df = pd.DataFrame([input_data])
    
    # ทำการ predict
    prediction = model.predict(input_data_df)
    
    # ส่งผลการ predict ไปแสดงใน template
    print(prediction[0])
    print(type(prediction[0]))

    return render_template('index.html', cluster=prediction[0])

if __name__ == '__main__':
    app.run(port = 3000, debug=True)