import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 設定頁面配置
st.set_page_config(page_title="CRISP-DM 線性迴歸分析", layout="wide")

# 標題
st.title("📊 CRISP-DM 線性迴歸分析系統")
st.markdown("**作業目標**：依照 CRISP-DM 方法論完成簡單線性迴歸分析")

# 側邊欄：參數設定
st.sidebar.header("🎛️ 參數設定")
st.sidebar.markdown("---")

# 使用者可調整的參數
a_true = st.sidebar.slider("斜率 a (真實值)", -10.0, 10.0, 2.5, 0.1)
b_true = st.sidebar.slider("截距 b (真實值)", -50.0, 50.0, 10.0, 1.0)
noise_level = st.sidebar.slider("噪音程度 (標準差)", 0.0, 20.0, 5.0, 0.5)
n_points = st.sidebar.slider("資料點數量", 20, 500, 100, 10)
random_seed = st.sidebar.number_input("隨機種子", 0, 9999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 真實模型")
st.sidebar.latex(f"y = {a_true:.2f}x + {b_true:.2f}")

# 建立分頁
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ 業務理解", "2️⃣ 資料理解", "3️⃣ 資料準備", 
    "4️⃣ 模型建立", "5️⃣ 模型評估", "6️⃣ 部署"
])

# ============================================================
# 階段 1: 業務理解 (Business Understanding)
# ============================================================
with tab1:
    st.header("1️⃣ 業務理解 (Business Understanding)")
    
    st.markdown("""
    ### 🎯 專案目標
    本專案目的是建立一個**簡單線性迴歸模型**，用於：
    - 理解兩個變數之間的線性關係
    - 預測因變數 y 基於自變數 x 的值
    - 評估模型的預測準確度
    
    ### 📝 問題定義
    給定一組資料點 (x, y)，我們想要找到最佳擬合直線：
    """)
    
    st.latex(r"y = ax + b")
    
    st.markdown("""
    其中：
    - **a**: 斜率 (slope)，表示 x 每增加 1 單位，y 的變化量
    - **b**: 截距 (intercept)，表示 x = 0 時 y 的值
    
    ### ✅ 成功標準
    - 模型能準確捕捉資料的線性趨勢
    - R² 分數 > 0.8 表示良好的擬合度
    - 殘差呈現隨機分布（無明顯模式）
    """)
    
    st.info("💡 **Prompt**: 請說明線性迴歸的業務應用場景，例如：銷售預測、成本估算、趨勢分析等")

# ============================================================
# 階段 2: 資料理解 (Data Understanding)
# ============================================================
with tab2:
    st.header("2️⃣ 資料理解 (Data Understanding)")
    
    # 生成合成資料
    np.random.seed(random_seed)
    X = np.random.uniform(-10, 10, n_points)
    noise = np.random.normal(0, noise_level, n_points)
    y = a_true * X + b_true + noise
    
    # 建立 DataFrame
    df = pd.DataFrame({'X': X, 'y': y})
    
    st.markdown("### 📊 資料集概覽")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("資料點數量", n_points)
    with col2:
        st.metric("特徵數量", 1)
    with col3:
        st.metric("目標變數", "y (連續)")
    
    st.markdown("### 🔍 資料預覽（前 10 筆）")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### 📈 統計摘要")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("### 📉 資料分布視覺化")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 散點圖
    axes[0].scatter(df['X'], df['y'], alpha=0.5, s=30)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('散點圖 (Scatter Plot)')
    axes[0].grid(True, alpha=0.3)
    
    # X 的分布
    axes[1].hist(df['X'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('頻率')
    axes[1].set_title('X 的分布')
    axes[1].grid(True, alpha=0.3)
    
    # y 的分布
    axes[2].hist(df['y'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('頻率')
    axes[2].set_title('y 的分布')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("💡 **Prompt**: 觀察資料的分布特性，是否有異常值？X 和 y 之間是否存在線性關係？")

# ============================================================
# 階段 3: 資料準備 (Data Preparation)
# ============================================================
with tab3:
    st.header("3️⃣ 資料準備 (Data Preparation)")
    
    st.markdown("""
    ### 🔧 資料處理步驟
    
    1. **資料清理**: 檢查缺失值和異常值
    2. **特徵工程**: 將 X 轉換為 sklearn 所需的 2D 陣列格式
    3. **資料分割**: 將資料分為訓練集和測試集
    """)
    
    # 檢查缺失值
    st.markdown("#### ✓ 缺失值檢查")
    missing_check = df.isnull().sum()
    st.dataframe(pd.DataFrame({
        '欄位': missing_check.index,
        '缺失數量': missing_check.values,
        '缺失比例(%)': (missing_check.values / len(df) * 100).round(2)
    }), use_container_width=True)
    
    # 資料轉換
    st.markdown("#### ✓ 資料轉換")
    X_prepared = X.reshape(-1, 1)  # sklearn 需要 2D 陣列
    y_prepared = y
    
    st.code("""
# 將 1D 陣列轉換為 2D 陣列（sklearn 要求）
X_prepared = X.reshape(-1, 1)
y_prepared = y
    """, language="python")
    
    st.markdown("#### ✓ 資料分割")
    test_size = st.slider("測試集比例", 0.1, 0.5, 0.2, 0.05)
    
    # 分割資料
    split_index = int(len(X_prepared) * (1 - test_size))
    indices = np.random.permutation(len(X_prepared))
    
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    X_train = X_prepared[train_indices]
    X_test = X_prepared[test_indices]
    y_train = y_prepared[train_indices]
    y_test = y_prepared[test_indices]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("訓練集大小", len(X_train))
    with col2:
        st.metric("測試集大小", len(X_test))
    
    st.success(f"✅ 資料準備完成！訓練集: {len(X_train)} 筆，測試集: {len(X_test)} 筆")
    
    st.info("💡 **Prompt**: 為什麼需要將資料分為訓練集和測試集？這對模型評估有什麼重要性？")

# ============================================================
# 階段 4: 模型建立 (Modeling)
# ============================================================
with tab4:
    st.header("4️⃣ 模型建立 (Modeling)")
    
    st.markdown("""
    ### 🤖 模型選擇
    選用 **線性迴歸 (Linear Regression)** 模型，使用最小平方法 (Ordinary Least Squares, OLS) 求解。
    
    ### 📐 數學原理
    目標是最小化殘差平方和：
    """)
    
    st.latex(r"\min_{a,b} \sum_{i=1}^{n} (y_i - (ax_i + b))^2")
    
    # 訓練模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    a_pred = model.coef_[0]
    b_pred = model.intercept_
    
    st.markdown("### 🎯 模型訓練結果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 預測參數")
        st.metric("斜率 (a)", f"{a_pred:.4f}")
        st.metric("截距 (b)", f"{b_pred:.4f}")
        st.latex(f"\\hat{{y}} = {a_pred:.4f}x + {b_pred:.4f}")
    
    with col2:
        st.markdown("#### 真實參數")
        st.metric("真實斜率 (a)", f"{a_true:.4f}")
        st.metric("真實截距 (b)", f"{b_true:.4f}")
        st.latex(f"y = {a_true:.4f}x + {b_true:.4f}")
    
    # 預測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 視覺化
    st.markdown("### 📊 模型視覺化")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 繪製訓練資料
    ax.scatter(X_train, y_train, alpha=0.5, label='訓練資料', s=40)
    ax.scatter(X_test, y_test, alpha=0.5, label='測試資料', s=40, color='orange')
    
    # 繪製預測線
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line_pred = model.predict(X_line)
    ax.plot(X_line, y_line_pred, 'r-', linewidth=2, label=f'預測線: y={a_pred:.2f}x+{b_pred:.2f}')
    
    # 繪製真實線
    y_line_true = a_true * X_line + b_true
    ax.plot(X_line, y_line_true, 'g--', linewidth=2, label=f'真實線: y={a_true:.2f}x+{b_true:.2f}')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('線性迴歸模型擬合結果', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    st.info("💡 **Prompt**: 觀察預測線和真實線的差異，噪音程度如何影響模型的擬合效果？")

# ============================================================
# 階段 5: 模型評估 (Evaluation)
# ============================================================
with tab5:
    st.header("5️⃣ 模型評估 (Evaluation)")
    
    st.markdown("### 📏 評估指標")
    
    # 計算評估指標
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 訓練集表現")
        st.metric("MSE (均方誤差)", f"{train_mse:.4f}")
        st.metric("RMSE (均方根誤差)", f"{train_rmse:.4f}")
        st.metric("R² (決定係數)", f"{train_r2:.4f}")
    
    with col2:
        st.markdown("#### 測試集表現")
        st.metric("MSE", f"{test_mse:.4f}")
        st.metric("RMSE", f"{test_rmse:.4f}")
        st.metric("R²", f"{test_r2:.4f}")
    
    # 指標說明
    st.markdown("""
    ### 📚 指標說明
    - **MSE (Mean Squared Error)**: 預測誤差的平方平均值，越小越好
    - **RMSE (Root Mean Squared Error)**: MSE 的平方根，與目標變數單位相同
    - **R² (R-squared)**: 模型解釋變異的比例，範圍 0-1，越接近 1 越好
    """)
    
    # 殘差分析
    st.markdown("### 🔍 殘差分析")
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 殘差分布圖
    axes[0].scatter(y_train_pred, residuals_train, alpha=0.5, label='訓練集')
    axes[0].scatter(y_test_pred, residuals_test, alpha=0.5, label='測試集', color='orange')
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('預測值')
    axes[0].set_ylabel('殘差')
    axes[0].set_title('殘差圖 (Residual Plot)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 殘差直方圖
    axes[1].hist(residuals_train, bins=30, alpha=0.6, label='訓練集', edgecolor='black')
    axes[1].hist(residuals_test, bins=30, alpha=0.6, label='測試集', color='orange', edgecolor='black')
    axes[1].set_xlabel('殘差')
    axes[1].set_ylabel('頻率')
    axes[1].set_title('殘差分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 評估結論
    st.markdown("### ✅ 評估結論")
    if test_r2 > 0.8:
        st.success(f"✅ 模型表現優秀！測試集 R² = {test_r2:.4f} > 0.8")
    elif test_r2 > 0.6:
        st.warning(f"⚠️ 模型表現尚可，測試集 R² = {test_r2:.4f}，建議調整參數或增加資料")
    else:
        st.error(f"❌ 模型表現不佳，測試集 R² = {test_r2:.4f}，需要重新檢視模型或資料")
    
    st.info("💡 **Prompt**: 理想的殘差圖應該呈現什麼樣的模式？如果殘差呈現系統性模式代表什麼？")

# ============================================================
# 階段 6: 部署 (Deployment)
# ============================================================
with tab6:
    st.header("6️⃣ 部署 (Deployment)")
    
    st.markdown("""
    ### 🚀 模型部署
    此應用程式已經是一個完整的部署範例，使用 **Streamlit** 框架建立。
    
    ### 💻 部署方式
    """)
    
    st.code("""
# 1. 安裝必要套件
pip install streamlit numpy pandas scikit-learn matplotlib seaborn

# 2. 執行應用程式
streamlit run app.py

# 3. 部署到雲端（可選）
# - Streamlit Cloud (免費): https://streamlit.io/cloud
# - Heroku
# - AWS / GCP / Azure
    """, language="bash")
    
    st.markdown("### 🎯 互動式預測")
    st.markdown("輸入新的 X 值，模型將預測對應的 y 值：")
    
    x_new = st.number_input("輸入 X 值", value=5.0, step=0.1)
    y_pred_new = model.predict([[x_new]])[0]
    y_true_new = a_true * x_new + b_true
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("輸入 X", f"{x_new:.2f}")
    with col2:
        st.metric("預測 y", f"{y_pred_new:.2f}")
    with col3:
        st.metric("真實 y (無噪音)", f"{y_true_new:.2f}")
    
    # 批量預測
    st.markdown("### 📊 批量預測")
    if st.button("生成預測報告"):
        pred_df = pd.DataFrame({
            'X': X_test.flatten(),
            'y_真實值': y_test,
            'y_預測值': y_test_pred,
            '殘差': y_test - y_test_pred,
            '絕對誤差': np.abs(y_test - y_test_pred)
        })
        pred_df = pred_df.sort_values('X').head(20)
        st.dataframe(pred_df, use_container_width=True)
        
        # 下載按鈕
        csv = pred_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載預測結果 (CSV)",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    
    st.markdown("### 📋 CRISP-DM 完整流程總結")
    st.markdown("""
    | 階段 | 主要任務 | 完成狀態 |
    |------|---------|---------|
    | 1. 業務理解 | 定義目標、問題和成功標準 | ✅ |
    | 2. 資料理解 | 探索資料、統計分析、視覺化 | ✅ |
    | 3. 資料準備 | 清理、轉換、分割資料 | ✅ |
    | 4. 模型建立 | 選擇演算法、訓練模型 | ✅ |
    | 5. 模型評估 | 計算指標、殘差分析 | ✅ |
    | 6. 部署 | 建立互動介面、提供預測服務 | ✅ |
    """)
    
    st.success("🎉 恭喜！你已完成完整的 CRISP-DM 線性迴歸專案！")
    
    st.info("💡 **Prompt**: 思考如何將此模型應用到真實場景，例如房價預測、銷售預測等？需要哪些額外的改進？")

# 頁尾
st.markdown("---")
st.markdown("**作業完成指標**: ✅ CRISP-DM 方法論 | ✅ 可調參數 | ✅ Web 框架 | ✅ 完整說明")
