import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel('2222222.xlsx')

# 提取特征和目标变量
X = df.drop('Unplanned reoperation', axis=1)
y = df['Unplanned reoperation']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'XGBoost': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# 训练模型并评估性能
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    results[name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision
    }

# 找到最佳模型
best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
best_model = models[best_model_name]

# 使用 SHAP 解释最佳模型
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

# 计算每个特征的 SHAP 平均值并按降序排序
feature_importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns).sort_values(
    ascending=False)

# 创建 Streamlit 应用
st.title('再手术风险预测模型')

# 显示模型性能
st.subheader('模型性能比较')
performance_df = pd.DataFrame(results).T
st.dataframe(performance_df)

# 显示最佳模型
st.subheader('最佳模型')
st.write(f'最佳模型是 {best_model_name}，基于 F1 Score 评估。')

# 显示 SHAP 汇总条形图
st.subheader('SHAP 汇总条形图')
fig, ax = plt.subplots()
shap.plots.bar(shap_values, max_display=len(X.columns), show=False)
st.pyplot(fig)

# 显示特征重要性排序
st.subheader('特征重要性排序')
st.write(feature_importance)

# 侧边栏 - 用户输入特征
st.sidebar.header('患者特征输入')
input_features = {}
for feature in X.columns:
    min_value = int(X[feature].min())
    max_value = int(X[feature].max())
    input_features[feature] = st.sidebar.slider(
        feature, min_value, max_value, (min_value + max_value) // 2)

# 预测按钮
if st.sidebar.button('预测再手术风险'):
    input_df = pd.DataFrame([input_features])
    prediction = best_model.predict(input_df)[0]
    prediction_proba = best_model.predict_proba(input_df)[0][1] if hasattr(best_model, 'predict_proba') else None

    st.subheader('预测结果')
    st.write(f'再手术风险预测: {"有风险" if prediction == 1 else "无风险"}')
    if prediction_proba is not None:
        st.write(f'再手术风险概率: {prediction_proba * 100:.2f}%')

    # 显示局部 SHAP 解释
    st.subheader('局部 SHAP 解释')
    local_shap_values = explainer(input_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(local_shap_values[0], show=False)
    st.pyplot(fig)