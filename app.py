from flask import Flask, render_template, request, send_file, redirect, url_for
import joblib
import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier, Pool
import joblib
import io
import os
import uuid

import google.generativeai as genai


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
app.config["UPLOAD_FOLDER"] = "temp"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["GET"])
def result():
    return render_template("resultPage.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/kmeans-analyze", methods=["GET", "POST"])
def kmeans_analyze():
    if request.method == "POST":
        file = request.files["file"]
        session_id = str(uuid.uuid4())

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return "ไฟล์ไม่รองรับ"

        df_pre = preprocess_for_kmeans(df)
        silhouette_scores = {}
        elbow_ks = list(range(1, 11))
        distortions = []

        for k in elbow_ks:
            model = KMeans(
                n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42
            )
            model.fit(df_pre)
            distortions.append(model.inertia_)

        for k in [3, 4]:
            model = KMeans(n_clusters=k, max_iter=50, random_state=42, n_init=10)

            labels = model.fit_predict(df_pre)
            silhouette_scores[k] = silhouette_score(df_pre, labels)

        # Save elbow graph
        plt.figure()
        plt.plot(elbow_ks, distortions, "bx-")
        plt.xlabel("k")
        plt.ylabel("Distortion")
        plt.title("Elbow Method")
        elbow_path = f"static/images/elbow_{session_id}.png"
        plt.savefig(elbow_path)
        plt.close()

        # Save raw file to temp
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}.csv")
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        df.to_csv(filepath, index=False)

        return render_template(
            "clustering/cluster_select.html",
            silhouette_scores=silhouette_scores,
            elbow_graph=elbow_path,
            session_id=session_id,
        )
    return render_template("clustering/upload_cluster.html")


@app.route("/kmeans-download", methods=["POST"])
def kmeans_download():
    num_clusters = int(request.form["num_clusters"])
    session_id = request.form["session_id"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}.csv")
    df = pd.read_csv(filepath)

    df_pre = preprocess_for_kmeans(df)

    model = KMeans(n_clusters=num_clusters, random_state=1)

    df["Cluster"] = model.fit_predict(df_pre)

    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(
        output, download_name=f"clustered_k{num_clusters}.xlsx", as_attachment=True
    )

@app.route("/catboost-train", methods=["GET", "POST"])
def catboost_train():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded", 400

        # Load file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file, dtype=str)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, dtype=str)
        else:
            return "Unsupported file type", 400

        if "Cluster" not in df.columns:
            return "Missing 'Cluster' column in uploaded data", 400

        # Split features & label
        X = df.drop(columns=["Cluster"])
        y = df["Cluster"]

        # ตรวจสอบ categorical columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = []
        for col in X.columns:
            try:
                pd.to_numeric(X[col].dropna().iloc[0])
                numeric_cols.append(col)
            except:
                pass
        categorical_cols = [col for col in categorical_cols if col not in numeric_cols]

        # แปลง numeric columns จริงๆ
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        for col in categorical_cols:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        # สร้าง Pool เพื่อรักษา categorical ให้ถูกต้อง
        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)

        # Train CatBoost model
        model = CatBoostClassifier(
            depth=4,
            iterations=200,
            l2_leaf_reg=1,
            learning_rate=0.01,
            verbose=False, 
            cat_features=categorical_cols
        )
        print("\n--- Debug ก่อนเข้าโมเดล CatBoost ---")
        print("Categorical Features:", categorical_cols)
        print("ตัวอย่างข้อมูล (X_train.head()):")
        print(X_train.head())
        print("dtype ของแต่ละฟีเจอร์ (X_train.dtypes):")
        print(X_train.dtypes)
        print("Categorical feature indices ใน Pool:")
        print(train_pool.get_cat_feature_indices())

        model.fit(train_pool)

        # Save model และข้อมูลที่จำเป็น
        model_id = str(uuid.uuid4())
        model_dir = "static/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/catboost_trained_{model_id}.pkl"
        joblib.dump(model, model_path)

        # บันทึก columns และ categorical values
        columns_path = f"{model_dir}/columns_{model_id}.json"
        with open(columns_path, "w") as f:
            json.dump(X.columns.tolist(), f)

        cat_values_map = {col: sorted(X[col].dropna().unique().tolist()) for col in categorical_cols}
        cat_values_path = f"{model_dir}/cat_values_{model_id}.json"
        with open(cat_values_path, "w") as f:
            json.dump(cat_values_map, f)

        # Save training data
        train_data_path = f"{model_dir}/train_data_{model_id}.csv"
        df.to_csv(train_data_path, index=False)

        # 🎉 Export model เป็น JSON ไว้ให้ Gemini ใช้งานต่อ
        model_json_path = f"{model_dir}/catboost_model_{model_id}.json"
        model.save_model(model_json_path, format="json")

        # Optional: Export tree เป็น text (.txt) สำหรับ Gemini เช่นเดิม (ถ้ายังต้องการใช้)
        graph = model.plot_tree(tree_idx=0, pool=train_pool)
        tree_text_path = f"{model_dir}/tree_text_{model_id}.txt"
        with open(tree_text_path, "w", encoding="utf-8") as f_out:
            f_out.write(graph.source)

        return redirect(url_for("predict_now", model_id=model_id))

    return render_template("train_catboost/upload_catboost.html")



@app.route("/download-catboost-model/<filename>")
def download_catboost_model(filename):
    return send_file(f"static/models/{filename}", as_attachment=True) 


@app.route("/predict-now/<model_id>", methods=["GET", "POST"])
def predict_now(model_id):
    import json

    model_path = f"static/models/catboost_trained_{model_id}.pkl"
    columns_path = f"static/models/columns_{model_id}.json"
    cat_values_path = f"static/models/cat_values_{model_id}.json"
    # tree_text_path = f"static/models/tree_text_{model_id}.txt"
    tree_text_path = f"static/models/catboost_model_{model_id}.json"

    # Load columns
    if not os.path.exists(columns_path):
        return "Feature column file not found", 404
    with open(columns_path, "r") as f:
        feature_columns = json.load(f)

    # Load categorical values
    cat_values = {}
    if os.path.exists(cat_values_path):
        with open(cat_values_path, "r") as f:
            cat_values = json.load(f)
    cat_features = list(cat_values.keys())

    # Load model
    model = joblib.load(model_path)

    # Load pre-rendered tree text
    if not os.path.exists(tree_text_path):
        return "Tree explanation file not found", 404
    # with open(tree_text_path, "r", encoding="utf-8") as f:
    #     dot_content = f.read()
    with open(tree_text_path, "r", encoding="utf-8") as f:
        dot_content = json.load(f)
    
    dot_content = json.dumps(dot_content)
    

    # Generate cluster explanation
    cluster_info = summarize_clusters_from_tree_with_gemini(dot_content, num_clusters=3)
    try:
        cluster_info_json = json.loads(cluster_info)
    except:
        cluster_info_json = None

    # Handle prediction
    result = None
    if request.method == "POST":
        input_data = {col: request.form[col] for col in feature_columns}
        df_input = pd.DataFrame([input_data])

        for col in df_input.columns:
            if col not in cat_features:
                try:
                    df_input[col] = pd.to_numeric(df_input[col])
                except:
                    pass

        result = model.predict(df_input)[0]

    return render_template(
        "predict/predict_now.html",
        model_id=model_id,
        columns=feature_columns,
        cat_values=cat_values,
        result=result,
        cluster_info=cluster_info_json
    )



@app.route("/", methods=["POST"])
def predict():

    num_clusters = request.form.get("num_clusters", "3")  # ค่าเริ่มต้นเป็น 3

    # ตรวจสอบว่ามาจากการกรอกข้อมูลคนเดียว
    if "single_submit" in request.form:
        # รับค่าจากฟอร์ม
        age = int(request.form["age"])
        unencoded_gender = request.form["gender"]
        # unencoded_gender = 'M'

        unencoded_education = request.form["education"]
        unencoded_marital = request.form["marital"]
        credit = float(request.form["credit"])
        trans_amt = float(request.form["trans_amt"])
        trans_count = float(request.form["trans_count"])
        months_on_book = float(request.form["MOB"])
        min_income = float(request.form["min-income"])
        max_income = float(request.form["max-income"])

        # สร้าง DataFrame สำหรับลูกค้าคนเดียว
        input_data = {
            "Age": age,
            "Gender": unencoded_gender,
            "Education_Level": unencoded_education,
            "Marital_Status": unencoded_marital,
            "Months_on_book": months_on_book,
            "Credit_Limit": credit,
            "Total_Trans_Amt": trans_amt,
            "Total_Trans_Count": trans_count,
            "Min_income": min_income,
            "Max_income": max_income,
        }
        data = pd.DataFrame([input_data])

    # ตรวจสอบว่ามาจากการอัปโหลดไฟล์
    elif "file_submit" in request.form and "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            # อ่านไฟล์ Excel
            data = pd.read_excel(file)

    else:
        return "Invalid input!", 400

    # โหลดโมเดลและพยากรณ์
    model_path = f"catboost_kprototypes_{num_clusters}clusters.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return f"Model for {num_clusters} clusters not found!", 500
    predictions = model.predict(data)

    # อ่านต้นไม้ที่เกี่ยวข้อง
    tree_path = f"Tree_B_kp{num_clusters}.txt"
    with open(tree_path, "r", encoding="utf-8") as f:
        tree_text = f.read()

    # ส่งให้ AI สรุปลักษณะกลุ่ม
    # cluster_description = summarize_clusters_from_tree(tree_text, int(num_clusters))
    # cluster_description = summarize_clusters_from_tree_with_gemini(
    #     tree_text, int(num_clusters)
    # )
    cluster_info = summarize_clusters_from_tree_with_gemini(
        tree_text, int(num_clusters)
    )
    cluster_info_json = json.loads(cluster_info)

    # เพิ่มคอลัมน์ Cluster ใน DataFrame
    data["Cluster"] = predictions

    # แปลง DataFrame เป็น HTML ตาราง
    tables = data.to_html(classes="data", header="true", index=False)

    # ส่งข้อมูลไปยัง template
    return render_template(
        "resultPage.html",
        tables=tables,
        cluster_info=cluster_info_json,
        # cluster_description=cluster_description
    )


def preprocess_for_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. แยกคอลัมน์ numerical และ categorical
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 2. Scaling เฉพาะ numerical columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 3. Factorize ค่าที่เป็น Categorical (ไม่ scale)
    for col in categorical_cols:
        df[col], _ = pd.factorize(df[col])

    return df


# ตั้งค่า API Key
genai.configure(api_key="AIzaSyAc1bcSbtjbzgorjVVTCiuxNxhsdJDPNO8")

# สร้าง Gemini Model
model = genai.GenerativeModel("gemini-2.0-flash-lite")


def summarize_clusters_from_tree_with_gemini(tree_text, num_clusters):
    prompt = f"""
คุณคือนักวิเคราะห์ข้อมูลลูกค้าด้วย AI

ฉันมีต้นไม้จากโมเดล CatBoost ที่ใช้ในการจำแนกลูกค้าออกเป็น {num_clusters} กลุ่ม (Cluster 0 ถึง {num_clusters - 1})
ต้นไม้จะแสดงการแบ่งด้วยฟีเจอร์ต่าง ๆ เช่น รายได้, วงเงิน, จำนวนครั้งที่ทำรายการ ฯลฯ

กรุณาช่วยสรุปลักษณะของแต่ละกลุ่ม (Cluster) ออกมาให้เข้าใจง่าย โดยตอบกลับมาในรูปแบบ JSON เท่านั้น ห้ามมีข้อความอื่นใดเพิ่มนอกเหนือจาก JSON เด็ดขาด และต้องเริ่มต้นด้วย '{{' และจบด้วย '}}' ตามนี้เท่านั้น:

{{
      "clusters": [
        {{
          "cluster": "Cluster 0",
          "group_name": "ชื่อของกลุ่ม (สั้นๆ 3-7 คำ)",
          "description": "ลักษณะของกลุ่มแบบสั้นๆ",
          "criteria": [
            "เกณฑ์ที่ 1 เช่น Credit Limit มากกว่า 3000",
            "เกณฑ์ที่ 2 เช่น Max Income ต่ำกว่า 50000"
          ]
        }},
        {{
          "cluster": "Cluster 1",
          "group_name": "ชื่อกลุ่ม",
          "description": "ลักษณะโดยรวมของกลุ่ม",
          "criteria": [
            "คุณสมบัติที่ใช้แบ่งกลุ่มที่ 1",
            "คุณสมบัติที่ใช้แบ่งกลุ่มที่ 2"
          ]
        }}
        // Cluster ต่อไป...
      ]
    }}

    ต้นไม้ที่ใช้แบ่งกลุ่ม:
    {tree_text}

    กรุณาส่งเฉพาะ JSON เท่านั้น ไม่ต้องมีข้อความอื่นใดเพิ่มเติม
"""

    response = model.generate_content(prompt)
    test_text = response.text.strip().replace("```", "")
    test_text = test_text.replace("json", "")
    print(test_text)
    return test_text if response.text else "ไม่ได้รับคำตอบจาก Gemini"


if __name__ == "__main__":
    app.run(port=3000, debug=True)
