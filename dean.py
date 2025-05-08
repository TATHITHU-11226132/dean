import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân tích & Dự đoán Doanh thu", layout="wide")

# Tiêu đề chính
st.title("Phân tích và Dự đoán Doanh thu Siêu thị")
st.markdown("**Trình bày bởi: [Tạ Thị Thư _ 11226132_CNTT&CDS_CLC_K64]** | Ngày: 09/05/2025")

# Thanh điều hướng
st.sidebar.title("Điều hướng")
section = st.sidebar.radio("Chọn phần trình bày:", [
    "1. Giới thiệu",
    "2. Khám phá dữ liệu",
    "3. Mô hình dự đoán",
    "4. Dự đoán doanh thu",
    "5. Kết luận và thảo luận"
])

# Hàm tải dữ liệu
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Sales-Superstore-Dataset.csv")
        return data
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None

# Hàm làm sạch dữ liệu
def clean_currency(val):
    try:
        val = str(val).replace('$', '').replace(',', '').replace('%', '').replace('(', '-').replace(')', '').strip()
        return float(val)
    except:
        return np.nan

# Tải và làm sạch dữ liệu
data = load_data()
if data is None:
    st.stop()
for col in ['Profit', 'Sales', 'Discount', 'Profit Ratio']:
    if col in data.columns:
        data[col] = data[col].apply(clean_currency)
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%m/%d/%Y')
data['Order Year'] = data['Order Date'].dt.year
data['Order Month'] = data['Order Date'].dt.month
data['Order Day'] = data['Order Date'].dt.day

# Chuẩn bị dữ liệu cho mô hình
features = ['Category', 'Region', 'Segment', 'Ship Mode', 'Sub-Category',
            'Discount', 'Profit', 'Quantity', 'Order Year', 'Order Month', 'Order Day']
target = 'Sales'
df_model = data[features + [target] + ['Order Date']].dropna()
X = df_model[features]
y = df_model[target]
cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(include='number').columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_features)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
order_dates_test = df_model.loc[X_test.index, 'Order Date']
test_indices = order_dates_test.argsort()
X_test = X_test.iloc[test_indices]
y_test = y_test.iloc[test_indices]
order_dates_test = order_dates_test.iloc[test_indices]

# Huấn luyện mô hình
pipeline_lr = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
pipeline_rf = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
pipeline_knn = Pipeline([('preprocessor', preprocessor), ('model', KNeighborsRegressor(n_neighbors=5))])
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_knn.fit(X_train, y_train)

# Phần 1: Giới thiệu
if section == "1. Giới thiệu":
    with st.container():
        st.header("1. Giới thiệu")
        st.markdown("""
        **Mục tiêu**: 
        - Phân tích dữ liệu doanh thu siêu thị để hiểu các yếu tố ảnh hưởng.
        - Dự đoán doanh thu dựa trên các đặc trưng như danh mục sản phẩm, khu vực, chiết khấu, v.v.
        - Ứng dụng: Hỗ trợ ra quyết định kinh doanh, tối ưu hóa chiến lược bán hàng.
        
        **Dữ liệu**: Bộ dữ liệu "Sales-Superstore-Dataset.csv" chứa thông tin về đơn hàng, doanh thu, lợi nhuận, và khách hàng từ 2011-2014.
        """)
        st.subheader("Thông tin dữ liệu")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- Số dòng: {len(data)}")
            st.write(f"- Số cột: {len(data.columns)}")
        with col2:
            st.write(f"- Thời gian: {data['Order Date'].min().year} - {data['Order Date'].max().year}")
            st.write(f"- Các cột chính: Sales, Profit, Discount, Category, Region, Segment")
        st.markdown("---")
        st.write("**Tiếp theo**: Hãy khám phá dữ liệu để hiểu xu hướng doanh thu!")

# Phần 2: Khám phá dữ liệu
elif section == "2. Khám phá dữ liệu":
    with st.container():
        st.header("2. Khám phá dữ liệu")
        st.markdown("Phân tích trực quan để hiểu xu hướng và đặc điểm của dữ liệu doanh thu.")
        
        # Biểu đồ cột: Tổng doanh số theo danh mục
        st.subheader("Tổng doanh số theo danh mục sản phẩm")
        sales_by_category = data.groupby("Category")["Sales"].sum().reset_index()
        fig_cat, ax_cat = plt.subplots(figsize=(8, 6))
        sns.barplot(x="Category", y="Sales", data=sales_by_category, palette="Blues_d", ax=ax_cat)
        ax_cat.set_title("Tổng doanh số theo danh mục sản phẩm", fontsize=14)
        ax_cat.set_xlabel("Danh mục sản phẩm", fontsize=12)
        ax_cat.set_ylabel("Tổng doanh số ($)", fontsize=12)
        ax_cat.tick_params(axis='x', rotation=45)
        st.pyplot(fig_cat)
        st.markdown("*Nhận xét*: Danh mục nào có doanh số cao nhất? Điều này gợi ý gì về chiến lược kinh doanh?")
        
        # Biểu đồ đường: Xu hướng doanh số theo thời gian
        st.subheader("Xu hướng doanh số theo thời gian")
        data["Month-Year"] = data["Order Date"].dt.to_period("M")
        sales_by_time = data.groupby("Month-Year")["Sales"].sum().reset_index()
        fig_time, ax_time = plt.subplots(figsize=(10, 6))
        ax_time.plot(sales_by_time["Month-Year"].astype(str), sales_by_time["Sales"], marker="o", color="b")
        ax_time.set_title("Xu hướng doanh số theo thời gian", fontsize=14)
        ax_time.set_xlabel("Tháng-Năm", fontsize=12)
        ax_time.set_ylabel("Tổng doanh số ($)", fontsize=12)
        ax_time.tick_params(axis='x', rotation=45)
        ax_time.grid(True)
        st.pyplot(fig_time)
        st.markdown("*Nhận xét*: Doanh số có xu hướng tăng hay giảm? Có mùa vụ nào đáng chú ý?")
        
        # Biểu đồ tròn: Phân phối doanh số theo phân khúc
        st.subheader("Phân phối doanh số theo phân khúc khách hàng")
        sales_by_segment = data.groupby("Segment")["Sales"].sum().reset_index()
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(sales_by_segment["Sales"], labels=sales_by_segment["Segment"], autopct="%1.1f%%",
                   startangle=140, colors=sns.color_palette("Set2"))
        ax_pie.set_title("Phân phối doanh số theo phân khúc khách hàng", fontsize=14)
        st.pyplot(fig_pie)
        st.markdown("*Nhận xét*: Phân khúc khách hàng nào đóng góp nhiều nhất vào doanh thu?")
        
        # Biểu đồ phân tán: Mối quan hệ giữa chiết khấu và lợi nhuận
        st.subheader("Mối quan hệ giữa chiết khấu và lợi nhuận")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Discount", y="Profit", hue="Category", size="Sales", data=data,
                        palette="deep", ax=ax_scatter)
        ax_scatter.set_title("Mối quan hệ giữa chiết khấu và lợi nhuận", fontsize=14)
        ax_scatter.set_xlabel("Chiết khấu", fontsize=12)
        ax_scatter.set_ylabel("Lợi nhuận ($)", fontsize=12)
        ax_scatter.legend(title="Danh mục")
        st.pyplot(fig_scatter)
        st.markdown("*Nhận xét*: Chiết khấu cao có làm giảm lợi nhuận không? Danh mục nào bị ảnh hưởng nhiều nhất?")
        
        # Biểu đồ heatmap: Doanh số theo khu vực và danh mục
        st.subheader("Doanh số theo khu vực và danh mục sản phẩm")
        pivot_table = data.pivot_table(values="Sales", index="Region", columns="Category",
                                      aggfunc="sum", fill_value=0)
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu",
                    cbar_kws={'label': 'Tổng doanh số ($)'}, ax=ax_heat)
        ax_heat.set_title("Doanh số theo khu vực và danh mục sản phẩm", fontsize=14)
        ax_heat.set_xlabel("Danh mục sản phẩm", fontsize=12)
        ax_heat.set_ylabel("Khu vực", fontsize=12)
        st.pyplot(fig_heat)
        st.markdown("*Nhận xét*: Khu vực nào có doanh số cao nhất theo danh mục?")
        
        # Biểu đồ hộp: Phân phối lợi nhuận theo phân khúc
        st.subheader("Phân phối lợi nhuận theo phân khúc khách hàng")
        fig_box, ax_box = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Segment", y="Profit", data=data, palette="Set3", ax=ax_box)
        ax_box.set_title("Phân phối lợi nhuận theo phân khúc khách hàng", fontsize=14)
        ax_box.set_xlabel("Phân khúc khách hàng", fontsize=12)
        ax_box.set_ylabel("Lợi nhuận ($)", fontsize=12)
        st.pyplot(fig_box)
        st.markdown("*Nhận xét*: Phân khúc nào có sự phân phối lợi nhuận ổn định nhất?")
        
        # Biểu đồ phân phối Sales
        st.subheader("Phân phối của Sales")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        sns.histplot(data['Sales'], bins=50, ax=ax_dist)
        ax_dist.set_title("Phân phối của Sales")
        st.pyplot(fig_dist)
        st.markdown("*Nhận xét*: Phân phối Sales có xu hướng nào nổi bật?")
        
        # Biểu đồ ma trận tương quan
        st.subheader("Ma trận tương quan")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(data[['Sales', 'Quantity', 'Discount', 'Profit', 'Order Year', 'Order Month', 'Order Day']].corr(),
                    annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title("Ma trận tương quan")
        st.pyplot(fig_corr)
        st.markdown("*Nhận xét*: Các biến nào có tương quan mạnh với Sales?")

# Phần 3: Mô hình dự đoán
elif section == "3. Mô hình dự đoán":
    with st.container():
        st.header("3. Mô hình dự đoán")
        st.markdown("Chúng ta đã huấn luyện 3 mô hình máy học để dự đoán doanh thu:")
        st.markdown("- **Linear Regression**: Mô hình tuyến tính đơn giản.")
        st.markdown("- **Random Forest**: Mô hình dựa trên cây quyết định, mạnh mẽ với dữ liệu phi tuyến.")
        st.markdown("- **K-nearest neighbors**: Mô hình dựa trên khoảng cách, phù hợp với các mẫu tương tự.")
        
        # Đánh giá mô hình
        st.subheader("Hiệu suất mô hình")
        y_pred_lr = pipeline_lr.predict(X_test)
        y_pred_rf = pipeline_rf.predict(X_test)
        y_pred_knn = pipeline_knn.predict(X_test)
        
        metrics = pd.DataFrame({
            "Mô hình": ["Linear Regression", "Random Forest", "KNN"],
            "R²": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_knn)],
            "MAE": [mean_absolute_error(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_knn)],
            "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lr)), np.sqrt(mean_squared_error(y_test, y_pred_rf)), np.sqrt(mean_squared_error(y_test, y_pred_knn))]
        })
        st.dataframe(metrics.style.format({"R²": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}))
        st.markdown("*Nhận xét*: Mô hình nào có hiệu suất tốt nhất? Tại sao?")
        
        # Biểu đồ Linear Regression: Dự đoán vs Thực tế
        st.subheader("Linear Regression: Dự đoán vs Thực tế")
        fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
        ax_lr.scatter(y_test, y_pred_lr, alpha=0.3, color='blue', label='Dự đoán (Linear Regression)')
        ax_lr.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Chuẩn')
        ax_lr.set_xlabel('Sales thực tế ($)')
        ax_lr.set_ylabel('Sales dự đoán ($)')
        ax_lr.set_title('Linear Regression: Dự đoán vs Thực tế')
        ax_lr.legend()
        ax_lr.grid(True)
        st.pyplot(fig_lr)
        st.markdown("*Nhận xét*: Các điểm gần đường chuẩn cho thấy dự đoán chính xác. Linear Regression phân bố thế nào?")
        
        # Biểu đồ Random Forest: Dự đoán vs Thực tế
        st.subheader("Random Forest: Dự đoán vs Thực tế")
        fig_rf, ax_rf = plt.subplots(figsize=(10, 6))
        ax_rf.scatter(y_test, y_pred_rf, alpha=0.3, color='green', label='Dự đoán (Random Forest)')
        ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Chuẩn')
        ax_rf.set_xlabel('Sales thực tế ($)')
        ax_rf.set_ylabel('Sales dự đoán ($)')
        ax_rf.set_title('Random Forest: Dự đoán vs Thực tế')
        ax_rf.legend()
        ax_rf.grid(True)
        st.pyplot(fig_rf)
        st.markdown("*Nhận xét*: Random Forest có cải thiện độ chính xác so với Linear Regression không?")
        
        # Biểu đồ KNN: Dự đoán vs Thực tế
        st.subheader("KNN: Dự đoán vs Thực tế")
        fig_knn, ax_knn = plt.subplots(figsize=(10, 6))
        ax_knn.scatter(y_test, y_pred_knn, alpha=0.3, color='red', label='Dự đoán (KNN)')
        ax_knn.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Chuẩn')
        ax_knn.set_xlabel('Sales thực tế ($)')
        ax_knn.set_ylabel('Sales dự đoán ($)')
        ax_knn.set_title('KNN: Dự đoán vs Thực tế')
        ax_knn.legend()
        ax_knn.grid(True)
        st.pyplot(fig_knn)
        st.markdown("*Nhận xét*: KNN hoạt động thế nào so với các mô hình khác? Có điểm nào bất thường không?")

# Phần 4: Dự đoán doanh thu
elif section == "4. Dự đoán doanh thu":
    with st.container():
        st.header("4. Dự đoán doanh thu")
        st.markdown("Hãy thử dự đoán doanh thu bằng cách nhập thông tin đơn hàng!")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Danh mục sản phẩm", data['Category'].unique())
            region = st.selectbox("Khu vực", data['Region'].unique())
            segment = st.selectbox("Phân khúc khách hàng", data['Segment'].unique())
            ship_mode = st.selectbox("Phương thức vận chuyển", data['Ship Mode'].unique())
            sub_category = st.selectbox("Danh mục phụ", data['Sub-Category'].unique())
        with col2:
            discount = st.number_input("Chiết khấu (0-1)", min_value=0.0, max_value=1.0, value=0.2)
            profit = st.number_input("Lợi nhuận", value=100.0)
            quantity = st.number_input("Số lượng", min_value=0, value=5)
            order_year = st.number_input("Năm đặt hàng", min_value=2000, max_value=2025, value=2018)
            order_month = st.number_input("Tháng đặt hàng", min_value=1, max_value=12, value=6)
            order_day = st.number_input("Ngày đặt hàng", min_value=1, max_value=31, value=15)
        
        model_choice = st.selectbox("Chọn mô hình:", ["Linear Regression", "Random Forest", "KNN"])
        
        if st.button("Dự đoán ngay!"):
            input_data = pd.DataFrame({
                'Category': [category], 'Region': [region], 'Segment': [segment],
                'Ship Mode': [ship_mode], 'Sub-Category': [sub_category],
                'Discount': [discount], 'Profit': [profit], 'Quantity': [quantity],
                'Order Year': [order_year], 'Order Month': [order_month], 'Order Day': [order_day]
            })
            try:
                if model_choice == "Linear Regression":
                    prediction = pipeline_lr.predict(input_data)[0]
                elif model_choice == "Random Forest":
                    prediction = pipeline_rf.predict(input_data)[0]
                else:
                    prediction = pipeline_knn.predict(input_data)[0]
                st.success(f"Doanh thu dự đoán: **${prediction:.2f}**")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

# Phần 5: Kết luận và thảo luận
elif section == "5. Kết luận và thảo luận":
    with st.container():
        st.header("5. Kết luận và thảo luận")
        st.markdown("""
        **Tóm tắt**:
        - Chúng ta đã phân tích dữ liệu doanh thu siêu thị, xác định các xu hướng quan trọng như doanh số theo danh mục, khu vực và thời gian.
        - Ba mô hình máy học (Linear Regression, Random Forest, KNN) được huấn luyện và đánh giá, với Random Forest thường cho kết quả tốt nhất.
        - Ứng dụng tương tác cho phép dự đoán doanh thu dựa trên các yếu tố đầu vào.
        
        **Ứng dụng thực tiễn**:
        - Hỗ trợ lập kế hoạch kinh doanh, tối ưu hóa chiết khấu và quản lý hàng tồn kho.
        - Định hướng chiến lược tiếp thị dựa trên phân khúc khách hàng và khu vực.
        
        **Hướng phát triển**:
        - Tích hợp thêm đặc trưng như hành vi khách hàng hoặc dữ liệu thời tiết.
        - Tối ưu hóa mô hình với các kỹ thuật nâng cao như Gradient Boosting.
        - Triển khai ứng dụng trên nền tảng đám mây để sử dụng thực tế.
        """)
        st.markdown("**Cảm ơn cô và các bạn đã theo dõi!**")