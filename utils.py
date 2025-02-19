import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

class PrepProcesor(): 
    # Đọc dữ liệu
    data = pd.read_csv("train.csv")

    # Xử lý giá trị thiếu
    imputer_num = SimpleImputer(strategy='median')
    imputer_cat = SimpleImputer(strategy='most_frequent')

    # Danh sách các cột
    num_features = ['Age', 'Fare']
    cat_features = ['Sex', 'Embarked', 'Pclass']

    # Mã hóa dữ liệu phân loại
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Chuẩn hóa dữ liệu số
    scaler = StandardScaler()

    # Tiền xử lý cột số và cột phân loại
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', imputer_num), ('scaler', scaler)]), num_features),
        ('cat', Pipeline([('imputer', imputer_cat), ('encoder', encoder)]), cat_features)
    ])

    # Tiền xử lý dữ liệu
    data_processed = preprocessor.fit_transform(data)

    # Xuất kết quả thành DataFrame mới
    processed_columns = num_features + list(encoder.get_feature_names_out(cat_features))
    data_cleaned = pd.DataFrame(data_processed, columns=processed_columns)

    # Chia dữ liệu theo tỷ lệ 70/15/15
    def split_data(data, train_size=0.7, valid_size=0.15, test_size=0.15):
        train_data, temp_data = train_test_split(data, test_size=(1 - train_size), random_state=42)
        valid_data, test_data = train_test_split(temp_data, test_size=(test_size / (valid_size + test_size)), random_state=42)
        return train_data, valid_data, test_data

    train_data, valid_data, test_data = split_data(data_cleaned)

    # Lựa chọn thuật toán Multiple Regression
    X_train = train_data.drop(columns=['Fare'])
    y_train = train_data['Fare']

    model = LinearRegression()
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    print("Cross Validation R2 Scores:", scores)
    print("Mean R2 Score:", scores.mean())

    # Lưu các tập dữ liệu
    train_data.to_csv("titanic_train.csv", index=False)
    valid_data.to_csv("titanic_valid.csv", index=False)
    test_data.to_csv("titanic_test.csv", index=False)
columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']