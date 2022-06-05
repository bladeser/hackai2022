# This is a sample Python script.
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def feed_func(description):

    if not description:
        return "Ошибка! Нужно ввести строку";
    if len(description) < 3:
        return "Ошибка! Строка слишком маленькая";
    if description.isnumeric():
        return "Ошибка! Строка должна содержать буквы (a-я) и числа (0-9)";
    if description.isdigit():
        return "Ошибка! Строка должна содержать буквы (a-я) и числа (0-9)";
    if description.isspace():
        return "Ошибка! Строка не может состоять из одних пробелов";

    description = description.lower();
    print(description)
    ml_model, tf_idf_vect = load_model()
    text_vecs = tf_idf_vect.transform([description])
    predict = ml_model.predict(text_vecs)
    #predict = "predict result is ready"
    print(predict)
    return predict


def text_preprocessing(string):
    if not string:
        return (False, "Ошибка! Нужно ввести строку")
    if len(string) < 3:
        return (False, "Ошибка! Строка слишком маленькая")
    if string.isnumeric():
        return (False, "Ошибка! Строка должна содержать буквы (a-я) и числа (0-9)")
    if string.isdigit():
        return (False, "Ошибка! Строка должна содержать буквы (a-я) и числа (0-9)")
    if string.isspace():
        return (False, "Ошибка! Строка не может состоять из одних пробелов")
    lower_case_text = string.lower()
    return (True, lower_case_text)


def load_model():
    """Load model structure from pickle dump"""
    model_filename = 'model.pkl'
    with open('server/model.pickle', 'rb') as file:
        model = pickle.load(file)
    with open('server/tf_idf_vect.pickle', 'rb') as file:
        tf_idf_vect = pickle.load(file)
    return model, tf_idf_vect


def post_tnved(string):
    url = "http://127.0.0.1:5001/tnved"
    payload = {
        "description": string}
    r = requests.post(url, json=payload)
    data = r.json()
    return data

def clear_text_only_letters(text_input):
    text_input = text_input.lower()
    cleared_text = re.sub('[^а-яa-z0-9]', ' ', text_input)
    #print(cleared_text)
    cleared_text = re.sub(' +', ' ', cleared_text)
    return cleared_text

def get_code(word):
    word = word.upper()
    kod_tnv = pd.read_csv('tnveddata_20211126.csv', sep=';', encoding='cp1251')
    #kod_tnv = pd.read_parquet('tnveddata_20211126.csv', sep=';', encoding='cp1251')
    kod_tnv['TNVED'] = kod_tnv['KOD_TNVED_SPR'].astype(str).map(lambda x: x[0:4])
    cnt_vec = CountVectorizer()
    cnt_vectors = cnt_vec.fit_transform(kod_tnv['OPISANIE_SPR'])
    word_vector = cnt_vec.transform([word])
    kod_tnv['sym'] = cosine_similarity(cnt_vectors, word_vector).reshape(-1, 1)
    kod_tnv = kod_tnv.sort_values('sym', ascending=False)
    if kod_tnv.iloc[0, 4] < 0.1:
        return ('Код не найден')
    print(kod_tnv.head())
    return (kod_tnv.iloc[0, 1], kod_tnv.iloc[0, 2])

def add_foplets(text_input):
    text = text_input.split()
    text_tri = list(map(lambda x: x[0:4], text))
    text = ' '.join(text)
    text_tri = ' '.join(text_tri)
    text = text + ' ' +  text_tri
    return text

def get_code2(word):
    #word = word.upper()
    word = clear_text_only_letters(word)
    word = add_foplets(word)
    #kod_tnv = pd.read_csv('tnveddata_20211126.csv', sep=';', encoding='cp1251')
    kod_tnv = pd.read_parquet('sprav.pq')
    #kod_tnv['TNVED'] = kod_tnv['KOD_TNVED_SPR'].astype(str).map(lambda x: x[0:4])
    cnt_vec = CountVectorizer()
    cnt_vectors = cnt_vec.fit_transform(kod_tnv['OPISANIE_SPR_CLEARED'])
    word_vector = cnt_vec.transform([word])
    kod_tnv['sym'] = cosine_similarity(cnt_vectors, word_vector).reshape(-1, 1)
    kod_tnv = kod_tnv.sort_values('sym', ascending=False)
    if kod_tnv.iloc[0, 3] < 0.1:
        return ('Код не найден')
    #print(kod_tnv.head())
    return (kod_tnv.iloc[0, 0], kod_tnv.iloc[0, 1])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #print(post_tnved('лошадь белая'))

    image = Image.open('pict5-72197342.png')
    col1, col2 = st.columns(2)
    col1.image(image, width=200)
    col2.header('Определение кода классификации товара методом Искусственного Интеллекта')

    st.title('Введите декларационное описание')
    title = st.text_input('Описание', '')
    if st.button('Получить код'):
        result_ans = post_tnved(title)
        str_input = str(result_ans['result'])[1:-1]
        proba = result_ans['proba']
        print(proba)
        #title = clear_text_only_letters(title)
        str_input2 = str(get_code2(title))
        #print('out', str_input)
        st.write('Наиболее вероятные коды: ')
        kod_tnv = pd.read_parquet('sprav.pq')
        kod_tnv['TNVED'] = kod_tnv['KOD_TNVED_SPR'].map(lambda x: str(x)[0:4])
        kod_tnv['OPISAN'] = kod_tnv['OPISANIE_SPR'].map(lambda x: ' '.join(x.split()[0:5]))
        #print(kod_tnv.head())
        print(str_input)
        kod_tnv = kod_tnv[kod_tnv['TNVED']==str_input]

        freq_opis = kod_tnv['OPISAN'].value_counts()
        #print(freq_opis)
        if freq_opis.shape[0] > 0:
            sample_str = freq_opis.index.values[0]
            print(sample_str)
        else:
            sample_str = 'сэмпл описания не найден'

        #print(freq_opis.index.values[0])
        if proba < 0.4:
            str_input = '****'
            sample_str = 'сэмпл описания не найден'


        st.write('Возможный код 1: ' + str_input2)
        st.write('Возможный код 2: ' + str_input + '. Cэмпл описания ' + sample_str)
    else:
        st.write('Введите описание товара')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
