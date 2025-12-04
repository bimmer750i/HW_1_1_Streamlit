import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sklearn
from local_preprocessing import local_preprocess

REQUIRED_COLUMNS = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

if 'form_inputs' not in st.session_state:
    st.session_state.form_inputs = {field: 0 for field in REQUIRED_COLUMNS}

st.markdown("<h2 style='text-align: center;'>Car price analysis</h2>",
            unsafe_allow_html=True)
# Кешируем модель
@st.cache_resource
def load_model():
    with open('best_elastic.pkl', 'rb') as file:
        return pickle.load(file)

# Кешируем Scaler
@st.cache_resource
def load_scaler():
    with open('norm_scaler.pkl', 'rb') as file:
        return pickle.load(file)

# Тут будет наша модель
model = load_model()

# Тут будет наш Scaler
scaler = load_scaler()

# Кешируем загруженный датасет
@st.cache_data
def load_dataset(file):
    return pd.read_csv(file,index_col=False)

# Кешируем отмасштабированный датасет с вещественными признаками
@st.cache_data
def get_scaled_numeric_dataset(df : pd.DataFrame, _scaler):
    return local_preprocess(df,_scaler)

# Кешируем датасет с вещественными признаками
@st.cache_data
def get_numeric_dataset(df : pd.DataFrame):
    return local_preprocess(df)

# Кешируем таргет
@st.cache_data
def get_target(df : pd.DataFrame):
    return df['selling_price']

# Кешируем график с распределением таргета
@st.cache_data
def get_target_distr(df_target : pd.DataFrame):
    fig_1, ax_1 = plt.subplots(figsize=(10, 10))
    ax_1.hist(df_target, bins=40)
    return fig_1

# Кешируем график с корреляциями
@st.cache_data
def get_corr(df : pd.DataFrame):
    fig_2, ax_2 = plt.subplots(figsize=(10, 10))
    correlation = df.corr()
    sns.heatmap(correlation, cmap='Blues', annot=True, ax=ax_2)
    return fig_2

# Кешируем график с попарными корреляциями
@st.cache_data
def get_pairplot(df : pd.DataFrame):
    pp = sns.pairplot(df)
    return pp

# Кешируем график с весами модели
@st.cache_resource
def get_weights():
    df_weights = pd.DataFrame(data=[model.coef_], columns=REQUIRED_COLUMNS)
    fig_3, ax_3 = plt.subplots(figsize=(10, 10))
    ax_3.bar(df_weights.columns, df_weights.iloc[0].values)
    ax_3.set_ylabel('Weight')
    ax_3.set_title('Model Feature Weights')
    return fig_3

# Импортируем датасет
with st.spinner("Loading dataset.."):
    df_train = load_dataset('df_train.csv')
st.success("DataSet loaded")

# Вещественные признаки
df_numeric = get_numeric_dataset(df_train)

# Наш таргет
target = get_target(df_train)

# Вот это я спросил у Deepseek - потому что хотел центрировать заголовок, но напрямую это не сделать,
# А HTML я не знаю(
st.markdown("<h2 style='text-align: center;'>EDA plots (scroll for more features)</h2>",
            unsafe_allow_html=True)

# Отобразим наш датасет
st.title("Dataset sample")
st.dataframe(df_train.sample(15))

# Распределение таргета
st.title("Target distribution")
target_distr = get_target_distr(target)
st.pyplot(target_distr)

# Матрица корреляций для вещественных признаков
st.title("Correlation Heatmap for numeric values")
corr = get_corr(df_numeric)
st.pyplot(corr)

# Распределения для численных величин
st.title("Pair plot for numeric values")
grid = get_pairplot(df_numeric)
st.pyplot(grid)

# Датасет для предсказаний
st.title("Download dataset for prediction. Tested on the dataset from Hometask 1")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df_test = load_dataset(uploaded_file)
    y_true = get_target(df_test)
    df_test = get_scaled_numeric_dataset(df_test, scaler)
    y_pred_test = model.predict(df_test)
    result_df = pd.DataFrame(
        data={"True": y_true, "Predicted": y_pred_test},
        columns=["True", "Predicted"])
    st.dataframe(result_df)

# Поля для ввода информации
# Сделал через форму - чтобы скрипт запускался заново только при нажатии кнопки
with st.form("prediction_form",clear_on_submit=False):
    cols = st.columns(len(REQUIRED_COLUMNS))
    inputs = {}
    for i, field in enumerate(REQUIRED_COLUMNS):
        with cols[i]:
            inputs[field] = st.number_input(
                label=field.replace('_', ' ').title(),
                min_value=0,
                value=0,
                step=1,
                key=f"form_{field}"
            )
    submitted = st.form_submit_button("Predict price")

    if submitted:
        obj = get_scaled_numeric_dataset(pd.DataFrame([inputs]),scaler)
        result = model.predict(obj)
        st.write(result)


st.title("Model weights")
weights = get_weights()
st.pyplot(weights)








