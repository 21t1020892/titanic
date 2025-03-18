import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import mlflow
import io
exp = mlflow.set_experiment("data_processing_experiment")
with mlflow.start_run(experiment_id=exp.experiment_id):
    def tien_xu_ly_du_lieu(updates_file=None):
        if updates_file is not None:
            df = pd.read_csv(updates_file)
        else:
            df = pd.read_csv("buoi2/data.txt")
        
        df = pd.read_csv("buoi2/data.txt")

        columns_to_drop = ["Cabin", "Ticket", "Name"]
        df.drop(columns=columns_to_drop, inplace=True)
        
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df.dropna(subset=['Embarked'], inplace=True)

        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
        df['Embarked'] = df['Embarked'].map({'Q': 0, 'S': 1, 'C': 2})

        scaler = StandardScaler()
        df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

        return df



    def test_train_size(actual_train_ratio, val_ratio_within_train, test_ratio):
        df = tien_xu_ly_du_lieu()
        X = df.drop(columns=['Survived'])
        y = df['Survived']
   
        # Chuyển đổi tỷ lệ phần trăm thành giá trị thực
        actual_train_size = actual_train_ratio / 100
        test_size = test_ratio / 100
        val_size = (val_ratio_within_train / 100) * actual_train_size  # Validation từ tập Train
        
        # Chia tập Train-Test trước
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        # Tiếp tục chia tập Train thành Train-Validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / actual_train_size, stratify=y_train, random_state=42)

        # Đảm bảo rằng số lần chia của KFold hợp lệ
        num_splits = max(2, int(1 / test_size))  # Đảm bảo n_splits >= 
        
        kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        
        
        return X_train, X_val, X_test, y_train, y_val, y_test, kf, df

    
    def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
        """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""
        
        # Chuyển đổi X_train, y_train sang NumPy array để tránh lỗi
        X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

        # Kiểm tra NaN hoặc Inf
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("Dữ liệu đầu vào chứa giá trị NaN!")
        if np.isinf(X_train).any() or np.isinf(y_train).any():
            raise ValueError("Dữ liệu đầu vào chứa giá trị vô cùng (Inf)!")

        # Chuẩn hóa dữ liệu để tránh tràn số
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
        m, n = X_train.shape
        #st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

        # Thêm cột bias (x0 = 1) vào X_train
        X_b = np.c_[np.ones((m, 1)), X_train]
        #st.write(f"Kích thước ma trận X_b: {X_b.shape}")

        # Khởi tạo trọng số ngẫu nhiên nhỏ
        w = np.random.randn(X_b.shape[1], 1) * 0.01  
        #st.write(f"Trọng số ban đầu: {w.flatten()}")

        # Gradient Descent
        for iteration in range(n_iterations):
            gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

            # Kiểm tra xem gradients có NaN không
            # st.write(gradients)
            if np.isnan(gradients).any():
                raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

            w -= learning_rate * gradients

        #st.success("✅ Huấn luyện hoàn tất!")
        #st.write(f"Trọng số cuối cùng: {w.flatten()}")
        return w



    # def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    #     """Huấn luyện hồi quy đa thức bằng Gradient Descent."""

    #     # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
    #     X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    #     y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    #     # Tạo đặc trưng đa thức
    #     poly = PolynomialFeatures(degree=degree, include_bias=False)
    #     X_poly = poly.fit_transform(X_train)

    #     # Chuẩn hóa dữ liệu để tránh tràn số
    #     scaler = StandardScaler()
    #     X_poly = scaler.fit_transform(X_poly)

    #     # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    #     m, n = X_poly.shape
    #     st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    #     # Thêm cột bias (x0 = 1)
    #     X_b = np.c_[np.ones((m, 1)), X_poly]
    #     st.write(f"Kích thước ma trận X_b: {X_b.shape}")

    #     # Khởi tạo trọng số ngẫu nhiên nhỏ
    #     w = np.random.randn(X_b.shape[1], 1) * 0.01  
    #     st.write(f"Trọng số ban đầu: {w.flatten()}")

    #     # Gradient Descent
    #     for iteration in range(n_iterations):
    #         gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

    #         # Kiểm tra nếu gradient có giá trị NaN
    #         if np.isnan(gradients).any():
    #             raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

    #         w -= learning_rate * gradients

    #     st.success("✅ Huấn luyện hoàn tất!")
    #     st.write(f"Trọng số cuối cùng: {w.flatten()}")
        
    #     return w, poly, scaler
    
    def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
        """Huấn luyện hồi quy đa thức **không có tương tác** bằng Gradient Descent."""

        # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
        X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

        # Tạo đặc trưng đa thức **chỉ thêm bậc cao, không có tương tác**
        X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])
        st.write(f"Kích thước ma trận X_poly: {X_poly[1]}")
        
        # Chuẩn hóa dữ liệu để tránh tràn số
        scaler = StandardScaler()
        X_poly = scaler.fit_transform(X_poly)

        # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
        m, n = X_poly.shape
        print(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

        # Thêm cột bias (x0 = 1)
        X_b = np.c_[np.ones((m, 1)), X_poly]
        print(f"Kích thước ma trận X_b: {X_b.shape}")

        # Khởi tạo trọng số ngẫu nhiên nhỏ
        w = np.random.randn(X_b.shape[1], 1) * 0.01  
        print(f"Trọng số ban đầu: {w.flatten()}")

        # Gradient Descent
        for iteration in range(n_iterations):
            gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

            # Kiểm tra nếu gradient có giá trị NaN
            if np.isnan(gradients).any():
                raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

            w -= learning_rate * gradients

        print("✅ Huấn luyện hoàn tất!")
        print(f"Trọng số cuối cùng: {w.flatten()}")
        
        return w

    def chon_mo_hinh(model_type, X_train, X_val, X_test, y_train, y_val, y_test, kf):
        """Chọn mô hình hồi quy tuyến tính bội hoặc hồi quy đa thức."""
        degree = 2
        fold_mse = []  # Danh sách MSE của từng fold
        scaler = StandardScaler()  # Chuẩn hóa dữ liệu cho hồi quy đa thức nếu cần

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
            X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            st.write(f"🚀 Fold {fold + 1}: Train size = {len(X_train_fold)}, Validation size = {len(X_valid)}")


            if model_type == "linear":
                w= train_multiple_linear_regression(X_train_fold, y_train_fold)
    
                w = np.array(w).reshape(-1, 1)
                
                X_valid = X_valid.to_numpy()


                X_valid_b = np.c_[np.ones((len(X_valid), 1)), X_valid]  # Thêm bias
                y_valid_pred = X_valid_b.dot(w)  # Dự đoán
            elif model_type == "polynomial":
                
                X_train_fold = scaler.fit_transform(X_train_fold)
                 
                w = train_polynomial_regression(X_train_fold, y_train_fold, degree)
                
                w = np.array(w).reshape(-1, 1)
                
                X_valid_scaled = scaler.transform(X_valid.to_numpy())
                X_valid_poly = np.hstack([X_valid_scaled] + [X_valid_scaled**d for d in range(2, degree + 1)])
                X_valid_b = np.c_[np.ones((len(X_valid_poly), 1)), X_valid_poly]
                
                y_valid_pred = X_valid_b.dot(w)  # Dự đoán
            else:
                raise ValueError("⚠️ Chọn 'linear' hoặc 'polynomial'!")

            mse = mean_squared_error(y_valid, y_valid_pred)
            fold_mse.append(mse)

            print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")

        # 🔥 Huấn luyện lại trên toàn bộ tập train
        if model_type == "linear":
            final_w = train_multiple_linear_regression(X_train, y_train)
            X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
            y_test_pred = X_test_b.dot(final_w)
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            final_w = train_polynomial_regression(X_train_scaled, y_train, degree)

            X_test_scaled = scaler.transform(X_test.to_numpy())
            X_test_poly = np.hstack([X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
            X_test_b = np.c_[np.ones((len(X_test_poly), 1)), X_test_poly]

            y_test_pred = X_test_b.dot(final_w)

        # 📌 Đánh giá trên tập test
        test_mse = mean_squared_error(y_test, y_test_pred)
        avg_mse = np.mean(fold_mse)  # Trung bình MSE qua các folds

        st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
        st.success(f"MSE trên tập test: {test_mse:.4f}")

        return final_w, avg_mse, scaler

    def bt_buoi3():
        uploaded_file = "buoi2/data.txt"
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")
        except FileNotFoundError:
            st.error("❌ Không tìm thấy tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
            st.stop()
        st.title("🔍 Tiền xử lý dữ liệu")
        
        st.subheader("📌 10 dòng đầu của dữ liệu gốc")
        st.write(df.head(10))
        
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

                    # Tạo báo cáo lỗi
        error_report = pd.DataFrame({
            'Cột': df.columns,
            'Giá trị thiếu': missing_values,
            'Outlier': [outlier_count.get(col, 0) for col in df.columns]
        })

                    # Hiển thị báo cáo lỗ
        st.table(error_report)

                    # Hiển thị số lượng dữ liệu trùng lặp
        st.write(f"🔁 **Số lượng dòng bị trùng lặp:** {duplicate_count}")         
        
        st.title("🔍 Tiền xử lý dữ liệu(Thực hành ở app tiền xử lý dữ liệu)")

        # Loại bỏ các cột không cần thiết
        st.subheader("1️⃣ Loại bỏ các cột không quan trọng")
        st.write("""
        Một số cột trong dữ liệu có thể không đóng góp nhiều vào kết quả dự đoán hoặc chứa quá nhiều giá trị thiếu. Việc loại bỏ các cột này giúp giảm độ phức tạp của mô hình và cải thiện hiệu suất.
        """)

        # Xử lý giá trị thiếu
        st.subheader("2️⃣ Xử lý giá trị thiếu")
        st.write("""
        Dữ liệu thực tế thường chứa các giá trị bị thiếu. Ta cần lựa chọn phương pháp thích hợp như điền giá trị trung bình, loại bỏ hàng hoặc sử dụng mô hình dự đoán để xử lý chúng nhằm tránh ảnh hưởng đến mô hình.
        """)

        # Chuyển đổi kiểu dữ liệu
        st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")
        st.write("""
        Một số cột trong dữ liệu có thể chứa giá trị dạng chữ (danh mục). Để mô hình có thể xử lý, ta cần chuyển đổi chúng thành dạng số bằng các phương pháp như one-hot encoding hoặc label encoding.
        """)

        # Chuẩn hóa dữ liệu số
        st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
        st.write("""
        Các giá trị số trong tập dữ liệu có thể có phạm vi rất khác nhau, điều này có thể ảnh hưởng đến độ hội tụ của mô hình. Ta cần chuẩn hóa dữ liệu để đảm bảo tất cả các đặc trưng có cùng trọng số khi huấn luyện mô hình.
        """)

        # Chia dữ liệu thành tập Train, Validation, và Test
        st.subheader("5️⃣ Chia dữ liệu thành tập Train, Validation, và Test")
        st.write("""
        Để đảm bảo mô hình hoạt động tốt trên dữ liệu thực tế, ta chia tập dữ liệu thành ba phần:
        - **Train**: Dùng để huấn luyện mô hình.
        - **Validation**: Dùng để điều chỉnh tham số mô hình nhằm tối ưu hóa hiệu suất.
        - **Test**: Dùng để đánh giá hiệu suất cuối cùng của mô hình trên dữ liệu chưa từng thấy.
        """)
        
        
        
        st.title("Lựa chọn thuật toán học máy: Multiple vs. Polynomial Regression")

        # Giới thiệu
        st.write("## 1. Multiple Linear Regression")
        st.write("""
        Hồi quy tuyến tính bội là một thuật toán học máy có giám sát, mô tả mối quan hệ giữa một biến phụ thuộc (output) và nhiều biến độc lập (input) thông qua một hàm tuyến tính.
        Ví dụ dự đoán giá nhà dựa trên diện tích, số phòng, vị trí, ... 
        
        Công thức tổng quát của mô hình hồi quy tuyến tính bội:
        """)
        st.image("buoi3/img1.png", caption="Multiple Linear Regression đơn", use_container_width =True)
        st.latex(r"""
        y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
        """)

        
    
    

        # Giới thiệu Polynomial Regression
        st.write("## 2. Polynomial Regression")

        st.write("Polynomial Regression mở rộng mô hình tuyến tính bằng cách thêm các bậc cao hơn của biến đầu vào.")
        
        st.image("buoi3/img3.png", caption="Polynomial Regression ", use_container_width =True)
        st.write("""
        Công thức tổng quát của mô hình hồi quy tuyến tính bội:
        """)
        st.latex(r"""
        y = w_0 + w_1x + w_2x^2 + w_3x^3 + \dots + w_nx^n
        """)

        
        st.write("""
        ### Hàm mất mát (Loss Function) của Linear Regression
        Hàm mất mát phổ biến nhất là **Mean Squared Error (MSE)**:
        """)
        st.latex(r"""
        MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        """)

        st.markdown(r"""
        Trong đó:
        - $n$: Số lượng điểm dữ liệu.
        - $y_i$: Giá trị thực tế của biến phụ thuộc.
        - $\hat{y}_i$: Giá trị dự đoán từ mô hình.
        """)

        st.markdown(r"""
        Mục tiêu của hồi quy tuyến tính bội là tìm các hệ số trọng số $w_0, w_1, w_2, ..., w_n$ sao cho giá trị MSE nhỏ nhất.

        ### Thuật toán Gradient Descent
        1. Khởi tạo các trọng số $w_0, w_1, w_2, ..., w_n$ với giá trị bất kỳ.
        2. Tính gradient của MSE đối với từng trọng số.
        3. Cập nhật trọng số theo quy tắc của thuật toán Gradient Descent.

        ### Đánh giá mô hình hồi quy tuyến tính bội
        - **Hệ số tương quan (R)**: Đánh giá mức độ tương quan giữa giá trị thực tế và giá trị dự đoán.
        - **Hệ số xác định (R²)**: Đo lường phần trăm biến động của biến phụ thuộc có thể giải thích bởi các biến độc lập:
        """)
        st.latex(r"""
        R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
        """)

        st.write("""
        - **Adjusted R²**: Điều chỉnh cho số lượng biến độc lập, giúp tránh overfitting:
        """)
        st.latex(r"""
        R^2_{adj} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k - 1} \right)
        """)

        st.markdown(r"""
        Trong đó:
        - $n$: Số lượng quan sát.
        - $k$: Số lượng biến độc lập.
        - $\bar{y}$: Giá trị trung bình của biến phụ thuộc.
        """)

        st.write("""
        
        - **Sai số chuẩn (SE)**: Đánh giá mức độ phân tán của sai số dự đoán quanh giá trị thực tế:
        """)
        st.latex(r"""
        SE = \sqrt{\frac{\sum (y_i - \hat{y}_i)^2}{n - k - 1}}
        """)

        st.write("""
        Các chỉ số này giúp đánh giá độ chính xác và khả năng khái quát hóa của mô hình hồi quy tuyến tính bội.
        """)
        # Vẽ biểu đồ so sánh
        st.write("## 3. Minh họa trực quan")

        # Tạo dữ liệu mẫu
        np.random.seed(0)
        x = np.sort(5 * np.random.rand(20, 1), axis=0)
        y = 2 * x**2 - 3 * x + np.random.randn(20, 1) * 2

        # Hồi quy tuyến tính
        lin_reg = LinearRegression()
        lin_reg.fit(x, y)
        y_pred_linear = lin_reg.predict(x)

        # Hồi quy bậc hai
        poly_features = PolynomialFeatures(degree=2)
        x_poly = poly_features.fit_transform(x)
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly, y)
        y_pred_poly = poly_reg.predict(x_poly)

        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, color='blue', label='Dữ liệu thực tế')
        ax.plot(x, y_pred_linear, color='red', label='Multiple Linear Regression')
        ax.plot(x, y_pred_poly, color='green', label='Polynomial Regression (bậc 2)')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        st.pyplot(fig)
        
        
        
        uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
    # Nếu có file upload, sử dụng nó, nếu không dùng file mặc định
        df = tien_xu_ly_du_lieu(uploaded_file)
        st.write(df.head(10))
    

        
        train_ratio = st.slider("Chọn tỷ lệ Train (%)", min_value=50, max_value=90, value=70, step=1)
        test_ratio = 100 - train_ratio  # Test tự động tính toán

        # Thanh kéo chọn tỷ lệ Validation trên Train
        val_ratio_within_train = st.slider("Chọn tỷ lệ Validation trong Train (%)", min_value=0, max_value=50, value=30, step=1)

        # Tính toán lại tỷ lệ Validation trên toàn bộ dataset
        val_ratio = (val_ratio_within_train / 100) * train_ratio
        actual_train_ratio = train_ratio - val_ratio

        # Hiển thị kết quả
        st.write(f"Tỷ lệ dữ liệu: Train = {actual_train_ratio:.1f}%, Validation = {val_ratio:.1f}%, Test = {test_ratio:.1f}%")

        X_train, X_val, X_test, y_train, y_val, y_test, kf, df  =test_train_size(actual_train_ratio, val_ratio_within_train,test_ratio)


        # Chọn mô hình    
        model_type = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])




        # Khi nhấn nút sẽ huấn luyện mô hình
        if st.button("Huấn luyện mô hình"):
        # Xác định model_type phù hợp
            #mlflow.log_param("model_type", model_type)
        
            model_type_value = "linear" if model_type == "Multiple Linear Regression" else "polynomial"

            # Gọi hàm với đúng thứ tự tham số
            final_w, avg_mse, poly, scaler= chon_mo_hinh(model_type_value, X_train, X_val, X_test, y_train, y_val, y_test, kf)


    
if __name__ == "__main__":
    bt_buoi3()