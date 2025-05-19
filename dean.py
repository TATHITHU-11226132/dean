import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Đọc dữ liệu UCI Heart Disease
data = pd.read_csv("heart_disease_uci.csv")

# Nhị phân hóa cột 'num' (0 = không bệnh, ≥1 = có bệnh)
data['HeartDisease'] = (data['num'] >= 1).astype(int)

# Tiền xử lý dữ liệu
@st.cache_data
def preprocess_data(data):
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = data[features]
    y = data['HeartDisease']

    # Mã hóa các cột phân loại
    categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Xử lý giá trị thiếu
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    binary_cols = ['fbs', 'exang']
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_binary = SimpleImputer(strategy='most_frequent')
    X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])
    X[binary_cols] = imputer_binary.fit_transform(X[binary_cols])

    # Chuẩn hóa các cột số
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, le_dict, imputer_numeric, imputer_binary, scaler

# Huấn luyện mô hình
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    return dt_model, knn_model, lr_model, kmeans, X_train, X_test, y_train, y_test, clusters

# Hàm vẽ ma trận nhầm lẫn
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Thực tế')
    ax.set_xlabel('Dự đoán')
    return fig

# Streamlit UI
st.title("Dự đoán Nguy cơ Bệnh Tim")
st.write("Nhập thông tin bệnh nhân và chọn mô hình để dự đoán nguy cơ mắc bệnh tim.")

# Tiền xử lý và huấn luyện
X, y, le_dict, imputer_numeric, imputer_binary, scaler = preprocess_data(data)
dt_model, knn_model, lr_model, kmeans, X_train, X_test, y_train, y_test, clusters = train_models(X, y)

# Form nhập liệu
with st.form("patient_form"):
    st.subheader("Thông tin bệnh nhân")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Tuổi", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Giới tính", ['Male', 'Female'])
        cp = st.selectbox("Loại đau ngực", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
        trestbps = st.number_input("Huyết áp nghỉ (mm Hg)", min_value=0.0, value=120.0)
    
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=0.0, value=200.0)
        fbs = st.selectbox("Đường huyết lúc đói (> 120 mg/dl)", ['TRUE', 'FALSE'])
        restecg = st.selectbox("Kết quả điện tâm đồ", ['normal', 'st-t abnormality', 'lv hypertrophy'])
        thalch = st.number_input("Nhịp tim tối đa", min_value=0.0, value=150.0)
    
    with col3:
        exang = st.selectbox("Đau thắt ngực do tập luyện", ['TRUE', 'FALSE'])
        oldpeak = st.number_input("ST chênh xuống (mm)", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Độ dốc đoạn ST", ['upsloping', 'flat', 'downsloping'])
        ca = st.number_input("Số mạch máu chính bị tắc", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Kết quả kiểm tra Thalium", ['normal', 'fixed defect', 'reversable defect'])
    
    # Thêm nút chọn mô hình
    model_choice = st.selectbox("Chọn mô hình dự đoán", ['Decision Tree', 'KNN', 'Logistic Regression'])
    
    submitted = st.form_submit_button("Dự đoán")

# Xử lý dự đoán
if submitted:
    # Tạo dữ liệu đầu vào
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [1 if fbs == 'TRUE' else 0],
        'restecg': [restecg],
        'thalch': [thalch],
        'exang': [1 if exang == 'TRUE' else 0],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Debug: Kiểm tra dữ liệu đầu vào ban đầu
    st.write("**Input Data (Before Preprocessing)**:")
    st.write(input_data)
    
    # Mã hóa dữ liệu phân loại
    for col in ['sex', 'cp', 'restecg', 'slope', 'thal']:
        try:
            input_data[col] = le_dict[col].transform(input_data[col])
        except ValueError as e:
            st.error(f"Error encoding column {col}: {e}")
            st.write(f"Valid values for {col}: {le_dict[col].classes_}")
            st.stop()
    
    # Xử lý giá trị thiếu và chuẩn hóa
    numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    binary_cols = ['fbs', 'exang']
    input_data[numeric_cols] = imputer_numeric.transform(input_data[numeric_cols])
    input_data[binary_cols] = imputer_binary.transform(input_data[binary_cols])
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # Đảm bảo tất cả các cột là số
    input_data = input_data.astype(float)
    
    # Debug: Kiểm tra dữ liệu sau khi tiền xử lý
    st.write("**Input Data (After Preprocessing)**:")
    st.write(input_data)
    
    # Debug: Kiểm tra kiểu dữ liệu
    st.write("**Data Types (After Preprocessing)**:")
    st.write(input_data.dtypes)
    
    # Dự đoán dựa trên mô hình được chọn
    st.subheader("Kết quả dự đoán")
    try:
        if model_choice == 'Decision Tree':
            pred = dt_model.predict(input_data)[0]
            st.write(f"**Cây quyết định**: {'Có nguy cơ bệnh tim' if pred == 1 else 'Không có nguy cơ bệnh tim'}")
        elif model_choice == 'KNN':
            pred = knn_model.predict(input_data)[0]
            st.write(f"**KNN**: {'Có nguy cơ bệnh tim' if pred == 1 else 'Không có nguy cơ bệnh tim'}")
        else:  # Logistic Regression
            pred = lr_model.predict(input_data)[0]
            prob = lr_model.predict_proba(input_data)[0, 1]
            st.write(f"**Hồi quy logistic**: {'Có nguy cơ bệnh tim' if pred == 1 else 'Không có nguy cơ bệnh tim'}")
            st.write(f"**Xác suất có nguy cơ bệnh tim**: {prob:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()
    
    # Trực quan hóa hiệu suất mô hình
    st.subheader("Hiệu suất mô hình")
    
    # Ma trận nhầm lẫn
    st.write("**Ma trận nhầm lẫn**")
    if model_choice == 'Decision Tree':
        cm_fig = plot_confusion_matrix(y_test, dt_model.predict(X_test), 'Ma trận nhầm lẫn - Cây quyết định')
        st.pyplot(cm_fig)
    elif model_choice == 'KNN':
        cm_fig = plot_confusion_matrix(y_test, knn_model.predict(X_test), 'Ma trận nhầm lẫn - KNN')
        st.pyplot(cm_fig)
    else:  # Logistic Regression
        cm_fig = plot_confusion_matrix(y_test, lr_model.predict(X_test), 'Ma trận nhầm lẫn - Hồi quy logistic')
        st.pyplot(cm_fig)
    
    # Đường cong ROC (chỉ cho Logistic Regression)
    if model_choice == 'Logistic Regression':
        st.write("**Đường cong ROC - Hồi quy logistic**")
        fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('Tỷ lệ dương tính giả')
        ax.set_ylabel('Tỷ lệ dương tính thật')
        ax.set_title('Đường cong ROC - Hồi quy logistic')
        ax.legend(loc='lower right')
        st.pyplot(fig)
    
    # Biểu đồ phân cụm PCA
    st.write("**Phân cụm bệnh nhân (PCA)**")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': clusters})
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in range(2):
        cluster_data = pca_df[pca_df['Cluster'] == cluster]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Nhóm {cluster}', alpha=0.6)
    ax.set_title('Phân cụm bệnh nhân (PCA)')
    ax.set_xlabel('Thành phần chính 1')
    ax.set_ylabel('Thành phần chính 2')
    ax.legend()
    st.pyplot(fig)