import streamlit as st
from captum.attr import LayerIntegratedGradients
from sklearn.metrics import f1_score

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertConfig
from collections import Counter

import torch
import re
import numpy as np

config = BertConfig.from_json_file("rubert-tiny-fakenews-finetuned/config.json")

model = AutoModelForSequenceClassification.from_pretrained(
    "rubert-tiny-fakenews-finetuned", config=config
)
tokenizer = AutoTokenizer.from_pretrained("rubert-tiny-fakenews-finetuned-tokenizer")

def predict_press_release(input_ids, token_type_ids, attention_mask):
    encoding = {
        'input_ids': input_ids.to(model.device),
        'token_type_ids': token_type_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device)
    }
    outputs = model(**encoding)
    return outputs

def predict(input_ids, token_type_ids, attention_mask):
    encoding = {
        'input_ids': input_ids.to(model.device),
        'token_type_ids': token_type_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device)
    }
    outputs = model(**encoding)
    return outputs

def squad_pos_forward_func(inputs, token_type_ids=None, attention_mask=None, position=0):
    pred = predict(inputs.to(torch.long), token_type_ids.to(torch.long), attention_mask.to(torch.long))
    pred = pred[position]
    return pred.max(1).values

lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

def get_description_interpreting(attrs):
    positive_weights = attrs
    return {
        'positive_weights': (
            positive_weights,
            {
                'min': np.min(positive_weights),
                'max': np.max(positive_weights)
            }
        ),
    }

def tokenize_data(text):
    return tokenizer(text['text'], padding=True, truncation=True, max_length=256, return_tensors='pt')

def clean(text):
    text = re.sub('[^а-яё ]', ' ', str(text).lower())
    text = re.sub(r" +", " ", text).strip()
    return text

def transform_token_ids(func_data, token_ids, word):
    tokens = list(map(lambda x: tokenizer.convert_ids_to_tokens([x])[0].replace('##', ''), tokenize_data({'text': clean(word)})['input_ids'][0]))
    weights = [func_data['positive_weights'][0][i] for i in token_ids]
    wts = []
    for i in range(len(weights)):
        if weights[i] > 0:
            #color = from_abs_to_rgb(func_data['positive_weights'][1]['min'], func_data['positive_weights'][1]['max'], weights[i])
            mn = max(func_data['positive_weights'][1]['min'], 0)
            mx = func_data['positive_weights'][1]['max']
            wts.append((weights[i] - mn)/ mx)
            #word = word.lower().replace(tokens[i], f'<span data-value="{(weights[i] - mn)/ mx}">{tokens[i]}</span>')
    try:
        if sum(wts) / len(wts) >= 0.2:
            return f'<span data-value={sum(wts) / len(wts)}>{word}</span>'
    except: pass
    return word

def build_text(tokens, func_data, current_text):
    splitted_text = current_text.split()
    splitted_text_iterator = 0
    current_word = ''
    current_word_ids = []
    for i, token in enumerate(tokens):
        decoded = tokenizer.convert_ids_to_tokens([token])[0]
        if decoded == '[CLS]': continue
        if not len(current_word):
            current_word = decoded
            current_word_ids.append(i)
        elif decoded.startswith('##'):
            current_word += decoded[2:]
            current_word_ids.append(i)
        else:
            while clean(splitted_text[splitted_text_iterator]) != current_word:
                splitted_text_iterator += 1
            current_word = decoded
            splitted_text[splitted_text_iterator] = transform_token_ids(func_data, current_word_ids, splitted_text[splitted_text_iterator])
            current_word_ids = []
    return ' '.join(splitted_text)

pattern = r'<span data-value=(.*?)>(.*?)</span>'

def highlight_words(text):
    def replace(match):
        value = match.group(1)
        word = match.group(2)
        color = f'rgba(255, 255, 0, {value})'  # Желтый цвет с прозрачностью, зависящей от значения
        return f'<span style="background-color:{color}; padding:2px; border-radius:4px; cursor:pointer" title="{value}">{word}</span>'
    return re.sub(pattern, replace, text)


label2id = {
    'true': 0,
    'fake': 1
}

id2label = {
    0: 'true',
    1: 'fake'
}


st.write("""
# Выявление дезинформации на основе модели трансформера
""")

user_text = st.text_area("Введите ваш текст здесь", height=200)

if 'btn_get_prediction' not in st.session_state:
    st.session_state.btn_get_prediction = False

if 'button_label_get_prediction' not in st.session_state:
    st.session_state.button_label_get_prediction = 'Получить предсказание'


def click_button():
    st.session_state.btn_get_prediction = not st.session_state.btn_get_prediction

    if st.session_state.btn_get_prediction:
        st.session_state.button_label_get_prediction = 'Назад'
    else:
        st.session_state.button_label_get_prediction = 'Получить предсказание'

st.button(st.session_state.button_label_get_prediction, on_click=click_button)

if st.session_state.btn_get_prediction:
    single_test = tokenizer(user_text, padding=True, truncation=True, max_length=512,
                            return_tensors='pt')

    predicted_class = np.argmax(predict_press_release(single_test['input_ids'], single_test['token_type_ids'], single_test['attention_mask']).logits.detach().numpy()[0])

    attrs = lig.attribute(single_test['input_ids'],
                          additional_forward_args=(single_test['attention_mask'], single_test['token_type_ids'], 0))

    attrs = np.array(list(map(lambda x: x.sum(), attrs[0])))

    descr = get_description_interpreting(attrs)

    st.write(id2label[Counter([predicted_class]).most_common()[0][0]])

    text = ''
    text += build_text(single_test['input_ids'][0], descr, user_text) + ' '

    st.markdown(highlight_words(text), unsafe_allow_html=True)