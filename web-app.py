import gc
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import seaborn as sns
import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, BertTokenizer
from urllib.request import urlopen
from PIL import Image

#daftar stopwords
STOPWORDS = {
    'CNN Indonesia --': '',
    'liputan6': '',
    'Liputan6.com': '',
    'kumparan.com': '',
    'Ayo share cerita pengalaman dan upload photo album travelingmu di sini. Silakan Daftar atau Masuk': '',
    'covid-19': 'covid',
    'Covid-19': 'Covid',
    'COVID-19': 'Covid',
    '[Gambas:Twitter]': '',
    '[Gambas:Instagram]': '',
    '[Gambas:Video CNN]': '',
}

#daftar keywords
KEYWORDS = {
    'covid',
    'corona',
    'korona',
    'bansos',
    'bantuan sosial',
    'psbb',
    'pembatasan sosial',
    'new normal',
    'normal baru',
    'normal yang baru',
    'vaksin',
    'vaksinasi',
    'odp',
    'orang dalam pengawasan',
    'pdp',
    'pasien dalam pengawasan',
    'otg',
    'orang tanpa gejala',
    'kontak erat',
    'suspect',
    'suspek',
    'sinovac',
    'astrazeneca',
    'moderna',
    'zona merah',
    'zona kuning',
    'zona hijau',
    'zona hitam',
    'isolasi',
    'karantina',
    'masker',
    'cuci tangan',
    'rapid test',
    'genose',
    'tes cepat',
    'swab',
    'wuhan',
    'wabah',
    'pandemi',
    'remdesivir',
    'observasi',
    'izin keramaian',
    'sars-cov',
    'lockdown',
    'wfh',
    'work from home',
    'pembatasan transportasi',
    'apd',
    'alat pelindung diri',
}

#menghapus karakter spesial dan url
def remove_special(text):
    text = text.replace('\\t', " ").replace(
        '\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'ignore').decode()
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

#menghapus stopword yang tidak diperlukan
def remove_stopwords(text):
    for key, replacement in STOPWORDS.items():
        text = text.replace(key, replacement)
        
    return text
    
#mengganti huruf kapital menjadi huruf kecil
def case_folding(text):
    return text.lower()

#menghapus angka
def remove_number(text):
    date_removed = re.sub(r"[(]\d*/\d*[)]", "", text)
    formatted_number_removed = re.sub(r"[0-9]{1,3}(.[0-9]{3})*(\,[0-9]+)*", "", date_removed)
    return re.sub(r'[0-9]', "", formatted_number_removed)

#menghapus tanda baca yang tidak diperlukan
def remove_punctuation(text, punctuation):
    return text.translate(str.maketrans(punctuation, ' '*len(punctuation)))

#menghapus spasi yang berlebih
def remove_multiple_spaces(text, punctuation):
    return ' '.join(word for word in text.split() if word not in punctuation)

#penggabungan semua fungsi preprocessing
def text_cleaning(text):
    stopwords_removed = remove_stopwords(text)
    case_folded = stopwords_removed.lower()
    number_removed = remove_number(case_folded)
    
    #tanda baca yang tidak diperlukan
    punctuation = '#$%*+<=>@[\\]^_{|}~'
    punctuation_removed = remove_punctuation(number_removed, punctuation)

    return punctuation_removed

#mengembalikan hasil encoding dari teks
def encode_text(text, max_len, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return {
        'text': text,
        'input_ids': encoding['input_ids'].flatten().unsqueeze(dim=0),
        'attention_mask': encoding['attention_mask'].flatten().unsqueeze(dim=0),
    }

#mengembalikan hasil encoding dari teks
def classify_text(model, encoded_text):    
    #karena prediksi tidak memerlukan backpropagation,
    #maka engine autograd dimatikan
    #untuk mengurangi konsumsi memori dan meningkatkan kecepatan pemrosesan
    with torch.no_grad(): 
        #output model
        output = model(
            input_ids=encoded_text['input_ids'],
            attention_mask=encoded_text['attention_mask']
        )
    
    #hasil prediksi kelas berdasarkan output model
    _, pred = torch.max(output, dim=1)

    #probabilitas masing-masing kelas
    #dihitung dengan menerapkan fungsi softmax pada output model
    probs = F.softmax(output, dim=1)

    return {
        'prediction': pred,
        'probabilities': probs, 
    }

def create_plot(position, class_names, probs):
    #dataframe prediksi
    pred_df = pd.DataFrame({
        'class_names': class_names,
        'values': probs
    })

    #inisiasi plot
    fig, ax = plt.subplots()
    
    #menampilkan persentase probabilitas prediksi
    #dalam bentuk barplot
    sns_ax = sns.barplot(x='values', y='class_names', data=pred_df, orient='h')

    #label sumbu y
    ax.set_ylabel('Kelas')

    #label sumbu x
    ax.set_xlabel('Probabilitas')

    #rentang nilai untuk sumbu x
    ax.set_xlim([0, 1])

    sns_ax.bar_label(sns_ax.containers[0])

    new_position = position.pyplot(fig)

    plt.clf()

    return new_position

def show_info(*args):
    for arg in args:
        arg.info('Memproses...')

#memuat tokenizer
@st.cache
def load_tokenizer():
    return BertTokenizer.from_pretrained('cahya/bert-base-indonesian-522M')

def check_state_dict(type):
    #nama state dictionary
    state_name = 'best_title_model_state.bin' if type == 'title' else 'best_content_model_state.bin'
    
    #url state dictionary
    state_url = st.secrets['title_state_url'] if type == 'title' else st.secrets['content_state_url']

    #download state dictionary jika belum ada di storage
    if not os.path.exists('./' + state_name):
        u = urlopen(state_url)
        data = u.read()
        u.close()

        with open(state_url, 'wb') as f:
            f.write(data)

#memuat state dictionary hasil pembelajaran
def load_state_dict_file(type):
    #nama state dictionary
    state_name = 'best_title_model_state.bin' if type == 'title' else 'best_content_model_state.bin'

    return torch.load(state_name, map_location=torch.device("cpu"))

@st.cache
def create_model(type):
    #instansiasi model klasifikasi
    model = NewsClassifier(5, 0.25)

    #memuat state_dict yang berisi nilai-nilai parameter
    #hasil pembelajaran
    state_dict = load_state_dict_file(type)

    #menggunakan nilai parameter hasil pelatihan pada model klasifikasi 
    model.load_state_dict(state_dict)

    #mengaktifkan evaluation flag
    model.eval()

    del state_dict

    return model

def analyze_framing(title_result_probs, content_result_probs):
    framed = 0
    not_framed = 0
    indecisive = 0

    for i in range(len(title_result_probs)):
        for j in range(len(content_result_probs)):
            if i != j:
                framed += (title_result_probs[i] * content_result_probs[j])

            if i == j and i != 4:
                not_framed += (title_result_probs[i] * content_result_probs[j])

            if i == j and i == 4:
                indecisive += (title_result_probs[i] * content_result_probs[j])
    
    
    prob = torch.tensor([framed, not_framed, indecisive])
    _, pred = torch.max(prob, dim=0)

    return {
        'prediction': pred,
        'probabilities': prob
    }

#kelas untuk instansiasi model klasifikasi
class NewsClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('cahya/bert-base-indonesian-522M')
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    #fungsi untuk melakukan forward pass
    def forward(self, input_ids, attention_mask):
        #output forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #pooled output
        pooled_output = outputs[1]

        #pooled output diproses oleh layer dropout
        output = self.drop(pooled_output)

        #setelah itu, hasilnya diproses oleh layer linear
        return self.out(output)

if __name__ == '__main__':
    #mengaktifkan garbage collector
    gc.enable()
    
    #set tampilan website dengan layout wide
    st.set_page_config(layout='wide')
    
    #menghapus whitespace pada bagian atas website
    st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

    
    st.write("# Deteksi Framing Pada Berita Online Covid-19 Berbahasa Indonesia")
    
    #membagi layout ke dalam dua kolom
    left, right = st.columns(2)

    #informasi program
    left.write("Dibuat Oleh Muhammad Razzaaq Fadilah - 140810180024")
    left.write("Dosen Pembimbing Pertama: Dr. Afrida Helen, MT")
    left.write("Dosen Pembimbing Kedua: Akmal, S.Si., MT")
    
    title_messages = left.empty()
    content_messages = left.empty()

    with left.form("input_form"):
        title = st.text_input("Masukkan Judul Berita", max_chars=256)
        content = st.text_area("Masukkan Isi Berita", max_chars=10000, height=1024)
        submitted = st.form_submit_button("Cek Framing")
    
    #nama-nama kelas untuk klasifikasi judul dan isi
    classification_class_names = ['Kesehatan', 'Politik', 'Ekonomi', 'Sosial', 'Lainnya']

    #nama-nama kelas untuk analisis framing
    framing_class_names = ['Ada Framing', 'Tidak Ada Framing', 'Tidak Dapat Diputuskan']

    right.write("Hasil Klasifikasi Judul Berita")
    title_fig_position = right.container()
    title_fig_position = create_plot(title_fig_position, classification_class_names, [0, 0, 0, 0, 0])

    right.write("Hasil Klasifikasi Isi Berita")
    content_fig_position = right.container()
    content_fig_position = create_plot(content_fig_position, classification_class_names, [0, 0, 0, 0, 0])

    right.write("Hasil Analisis Framing pada Berita")
    framing_fig_position = right.container()
    framing_fig_position = create_plot(framing_fig_position, framing_class_names, [0, 0, 0])

    if submitted:
        #cek apakah judul atau isi kosong
        empty_title_check = title.strip() == ''
        empty_content_check = content.strip() == ''

        if empty_title_check:
            title_messages.error('Judul berita tidak boleh kosong!')
            
        if empty_content_check:
            content_messages.error('Isi berita tidak boleh kosong!')
                
        if empty_title_check or empty_content_check:
            st.stop()
        
        #prepocessing teks input
        preprocessed_title = text_cleaning(title)
        preprocessed_content = text_cleaning(content)
        
        #cek apakah teks berkaitan dengan covid-19
        title_related_check = any(word in preprocessed_title for word in KEYWORDS)
        content_related_check = any(word in preprocessed_content for word in KEYWORDS)
        
        if not title_related_check:
            title_messages.error('Judul berita tidak terkait Covid-19!')

        if not content_related_check:
            content_messages.error('Isi berita tidak terkait Covid-19!')

        if not title_related_check or not content_related_check:
            st.stop()

        #memuat tokenizer yang digunakan
        tokenizer = load_tokenizer()

        #cek state dictionary pada storage
        check_state_dict('title')
        check_state_dict('content')
        
        #membuat model menggunakan state dictionary hasil pembelajaran
        title_model = create_model('title')
        content_model = create_model('content')

        #melakukan encoding pada teks
        encoded_title = encode_text(preprocessed_title, 30, tokenizer)
        encoded_content = encode_text(preprocessed_content, 512, tokenizer)

        #melakukan klasifikasi pada judul
        title_result = classify_text(title_model, encoded_title)
        content_result = classify_text(content_model, encoded_content)

        #mengubah dimensi hasil probabilitas
        title_result['probabilities'] = title_result['probabilities'].squeeze()
        content_result['probabilities'] = content_result['probabilities'].squeeze()

        #analisis framing
        framing_result = analyze_framing(title_result['probabilities'], content_result['probabilities'])
       
        create_plot(title_fig_position, classification_class_names, title_result['probabilities'])
        create_plot(content_fig_position, classification_class_names, content_result['probabilities'])
        create_plot(framing_fig_position, framing_class_names, framing_result['probabilities'])

        #menghapus objek agar tidak memenuhi memori
        del encoded_title
        del encoded_content
        del title_result
        del content_result
        del framing_result

        #menjalankan garbage collector
        gc.collect()