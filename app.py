import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
# Tải dữ liệu
def phan_gioi_thieu():
    uploaded_file = "titanic.csv"
    try:
        df = pd.read_csv(uploaded_file, delimiter=",")
    except FileNotFoundError:
        st.error("❌ Không tìm thấy tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
        st.stop()
    st.title("🔍 Tiền xử lý dữ liệu")

    # Hiển thị dữ liệu gốc
    st.subheader("📌 Hiển thị dữ liệu gốc")
    num_rows = st.number_input("Nhập số dòng muốn hiển thị:", min_value=1, max_value=len(df), value=10, step=1)
    st.write(df.head(num_rows))    
    st.subheader("🚨 Kiểm tra lỗi dữ liệu")
                # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()
                # Kiểm tra dữ liệu trùng lặp
    duplicate_count = df.duplicated().sum()
                # Kiểm tra giá trị quá lớn (outlier) bằng Z-score
    outlier_count = {
        col: (abs(zscore(df[col], nan_policy='omit')) > 3).sum()
        for col in df.select_dtypes(include=['number']).columns
    }

    error_report = pd.DataFrame({
        'Cột': df.columns,
        'Giá trị thiếu': missing_values,
        'Outlier': [outlier_count.get(col, 0) for col in df.columns]
    })
    st.table(error_report)
    st.write(f"🔁 **Số lượng dòng bị trùng lặp:** {duplicate_count}")      
    st.write(len(df))     
    st.header("⚙️ Các bước chính trong tiền xử lý dữ liệu")
    st.subheader("1️⃣ Loại bỏ các cột không cần thiết")
    st.write("""
        Một số cột trong dữ liệu có thể không ảnh hưởng đến kết quả dự đoán hoặc chứa quá nhiều giá trị thiếu.
        Bạn có thể chọn các cột cần loại bỏ dưới đây:
    """)
    columns_to_drop = st.multiselect("Chọn các cột muốn xóa:", df.columns)
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        st.success(f"Đã xóa các cột: {', '.join(columns_to_drop)}")
    st.subheader("📌 Dữ liệu sau khi loại bỏ cột:")
    st.write(df.head())
    st.subheader("2️⃣ Xử lý giá trị thiếu")
    st.write("""
        Dữ liệu thực tế thường có giá trị bị thiếu. Chúng ta cần xử lý để tránh ảnh hưởng đến mô hình.
        Bạn có thể chọn cách điền giá trị thiếu bằng **Trung bình (Mean)** hoặc **Trung vị (Median)**.
    """)

    age_option = st.radio("Chọn phương pháp điền giá trị thiếu cho 'Age':", ("Mean", "Median"))
    if age_option == "Mean":
        df["Age"].fillna(df["Age"].mean(), inplace=True)
    else:
        df["Age"].fillna(df["Age"].median(), inplace=True)

    st.write("Cột 'Embarked' có số lượng giá trị thiếu ít (2 dòng), sẽ bị xóa.")
    df.dropna(subset=["Embarked"], inplace=True)

    st.success("✅ Đã xử lý giá trị thiếu!")

    # Hiển thị dữ liệu sau khi xử lý giá trị thiếu
    st.subheader("📌 Dữ liệu sau khi xử lý giá trị thiếu:")
    st.write(df.head(10))

    st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")
    st.write("""
        Trong dữ liệu, có một số cột chứa giá trị dạng chữ (category). Ta cần chuyển đổi thành dạng số để mô hình có thể xử lý.
        - **Cột "Sex"**: Chuyển thành 1 (male), 0 (female).
        - **Cột "Embarked"**:   Chuyển thành 1 (Q), 2 (S), 3 (C).
        ```python
            df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Mã hóa giới tính
         
            df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})

        ```
        """)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})  # Mã hóa giới tính
    # df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-Hot Encoding
    df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})

    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
        Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình. Ta sẽ chuẩn hóa "Age" và "Fare" về cùng một thang đo bằng StandardScaler.
        
        ```python
            scaler = StandardScaler()
            df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

        ```
        """)
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
    st.write("Dữ liệu sau khi xử lý:")
    st.write(df.head(10))
    df.to_csv("processed_titanic.csv", index=False)  # Lưu mà không kèm chỉ mục
    st.subheader("5️⃣ Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
        ### 📌 Chia tập dữ liệu
        Bạn có thể chọn tỷ lệ dữ liệu dành cho **Train**, phần còn lại sẽ được chia đều cho **Validation** và **Test**.
    """)
    # Người dùng chọn tỷ lệ Train (70% mặc định)
    train_size = st.slider("Chọn phần trăm dữ liệu dùng để Train:", min_value=50, max_value=80, value=70, step=5)
    # Tính toán tỷ lệ Validation & Test
    val_test_size = (100 - train_size) / 2 / 100  # Chia đều cho Validation và Test
    train_size /= 100  # Chuyển train_size về dạng thập phân
    # Chia dữ liệu
    X = df.drop(columns=["Survived"])  # Biến đầu vào
    y = df["Survived"]  # Nhãn
    # Chia dữ liệu thành Train và (Validation + Test)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=(1 - train_size), stratify=y, random_state=42
    )
    # Chia (Validation + Test) thành Validation và Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    # Hiển thị kết quả chia
    st.write(f"📊 **Tỷ lệ chia dữ liệu:**")
    st.write(f"- **Train**: {train_size*100:.0f}% ({len(X_train_full)} mẫu)")
    st.write(f"- **Validation**: {val_test_size*100:.1f}% ({len(X_val)} mẫu)")
    st.write(f"- **Test**: {val_test_size*100:.1f}% ({len(X_test)} mẫu)")
    st.success("✅ Dữ liệu đã được chia thành công!")
    return  X_train_full, X_val, X_test, y_train_full, y_val, y_test
def phan_train(X_train, y_train, X_val, y_val, X_test, y_test):
    st.title("🚀 Huấn luyện mô hình")
    st.subheader(" mô hình Random Forest")
    st.write(f"""
        Mô hình Random Forest là một mô hình mạnh mẽ và linh hoạt, thường được sử dụng trong các bài toán phân loại và hồi quy.
        Ưu điểm:   
        - Xử lý tốt với dữ liệu lớn.
        - Không yêu cầu chuẩn hóa dữ liệu.
        - Dễ dàng xử lý overfitting.
        Nhược điểm:
        - Không hiệu quả với dữ liệu có nhiều giá trị thiếu.
        - Mất hiệu suất khi số lượng cây lớn.
        - Không thể hiển thị quá trình học.
        
        Chúng ta sẽ sử dụng mô hình Random Forest để dự đoán khả năng sống sót trên tàu Titanic.
        ```python
            from sklearn.ensemble import RandomForestClassifier

            # Khởi tạo mô hình
            model = RandomForestClassifier(random_state=42)

            # Huấn luyện mô hình
            model.fit(X_train, y_train)

        ```
        """)
    # Khởi tạo mô hình
    model = RandomForestClassifier(random_state=42)
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    st.write("🎯 Đánh giá mô hình bằng Cross-Validation")
    st.markdown("""
    Chúng ta có thể sử dụng `cross_val_score` từ `sklearn.model_selection`:

    ```python
    from sklearn.model_selection import cross_val_score

    # Đánh giá mô hình bằng cross-validation (5-Fold CV)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    ```
    
    """)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    st.write(f"Cross-validation scores: {cv_scores}")
    st.write(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    valid_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    st.write(f"✅ Validation Accuracy: {valid_acc:.4f}")
    st.write(f"✅ Test Accuracy: {test_acc:.4f}")
    return model, valid_acc, test_acc
def test_model(model):
    df = pd.read_csv("processed_titanic.csv")
    st.write("### Kiểm tra mô hình với giá trị nhập vào")
    # 1️⃣ Liệt kê các cột của DataFrame
    feature_columns = df.drop(columns=["Survived"]).columns
    st.write("🔹 Các cột đầu vào:", feature_columns.tolist())
    # 2️⃣ Tạo input cho từng cột
    input_data = {}
    for col in feature_columns:
        input_data[col] = st.number_input(f"Nhập giá trị cho {col}", value=0.0)
    # 3️⃣ Chuyển thành DataFrame
    input_df = pd.DataFrame([input_data])
    # 4️⃣ Dự đoán với model
    if st.button("Dự đoán"):
        prediction = model.predict(input_df)
        result = "🛑 Không sống sót" if prediction[0] == 0 else "✅ Sống sót"
        st.success(f"🔮 Dự đoán kết quả: {result}")

def report():
    X_train, X_val, X_test, y_train, y_val, y_test = phan_gioi_thieu()
    model, valid_acc, test_acc = phan_train(X_train, y_train, X_val, y_val, X_test, y_test)
    test_model(model)
if __name__ == "__main__":
    report()