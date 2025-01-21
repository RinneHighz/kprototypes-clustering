from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    return render_template('resultPage.html')



@app.route('/', methods=['POST'])
def predict():
    # รับค่าจาก form
    age = int(request.form['age'])
    unencoded_gender = request.form['gender']
    unencoded_education = request.form['education']
    unencoded_marital = request.form['marital']
    credit = float(request.form['credit'])
    trans_amt = float(request.form['trans_amt'])
    trans_count = float(request.form['trans_count'])
    months_on_book = float(request.form['MOB'])
    min_income = float(request.form['min-income'])
    max_income = float(request.form['max-income'])
    
    # เข้ารหัสค่าสตริงให้เป็นตัวเลข
    # gender = 1 if unencoded_gender == 'F' else 0
    
    gender = unencoded_gender 
    
    # education_mapping = {'High School': 0, 'Graduate': 1, 'Uneducated': 2, 'Unknown': 3, 'College': 4, 'Post-Graduate': 5, 'Doctorate': 6}
    # education = education_mapping[unencoded_education]
    education = unencoded_education
    
    marital = unencoded_marital
    
    # โหลดโมเดลจากไฟล์
    # model = joblib.load('decision_tree_kprototypes_3clusters.pkl')
    model = joblib.load('catboost_kprototypes_3clusters.pkl')

    
    # สร้าง input data สำหรับ prediction พร้อมชื่อคุณลักษณะ
    input_data = {
        'Age': age,
        'Gender': gender,
        'Education_Level': education,
        'Marital_Status': marital,
        'Months_on_book': months_on_book,
        'Credit_Limit': credit,
        'Total_Trans_Amt': trans_amt,
        'Total_Trans_Count': trans_count,
        'Min_income': min_income,
        'Max_income': max_income,
    }
    
    input_data_df = pd.DataFrame([input_data])
    
    # ทำการ predict
    prediction = model.predict(input_data_df)
    
    # ส่งผลการ predict ไปแสดงใน template
    print(prediction[0])
    print(type(prediction[0]))

    return render_template('resultPage.html', cluster=prediction[0])

if __name__ == '__main__':
    app.run(port = 3000, debug=True)