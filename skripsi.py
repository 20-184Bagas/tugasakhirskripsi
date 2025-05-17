import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import chi2
from scipy.stats import chi2 as chi2_table
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import nltk
from imblearn.over_sampling import SMOTE
from matplotlib.patches import Patch

# === NORMALISASI KAMUS ===
@st.cache_data
def load_normalization_dict():
    url = "https://raw.githubusercontent.com/20-184Bagas/tugasakhirskripsi/refs/heads/main/Kamus_Normalization-_2_%20(1).csv"
    df_norm = pd.read_csv(url)
    return dict(zip(df_norm['Tidak Baku'], df_norm['Baku']))

normalization_dict = load_normalization_dict()

# === FUNGSI DATA & PREPROCESSING ===
def load_data():
    url = pd.read_csv("https://raw.githubusercontent.com/20-184Bagas/tugasakhirskripsi/refs/heads/main/data-ulasan-wisata-madura.csv") 
    return url

def normalize_text(text):
    return " ".join([normalization_dict.get(w, w) for w in text.split()])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[{}]".format(string.punctuation), " ", text)
    text = re.sub(r"\\d+", "", text)
    stop_words = set(stopwords.words("indonesian"))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    normalized = normalize_text(" ".join(tokens))
    return normalized

def load_preprocessed_data():
    df = pd.read_csv("https://raw.githubusercontent.com/20-184Bagas/tugasakhirskripsi/refs/heads/main/hasil_preprocessing.csv")
    df["Label"] = df["Label"].str.lower() 
    label_mapping = {"positif": 1, "negatif": -1, "netral": 0}
    df["Encoded Label"] = df["Label"].map(label_mapping)
    df["Ulasan"] = df["Ulasan"].astype(str)
    return df 

# === SELEKSI FITUR ===
def chi_square_selection(count_df, y, alpha=0.001):
    scores, _ = chi2(count_df, y)
    df_chi = pd.DataFrame({"Fitur": count_df.columns, "Chi2": scores})
    chi_crit = chi2_table.ppf(1 - alpha, df=len(set(y)) - 1)
    selected = df_chi[df_chi["Chi2"] >= chi_crit]["Fitur"].tolist()
    return selected

def info_gain_selection(X_df, y, threshold=0.001):
    def entropy(vec):
        probs = vec.value_counts(normalize=True)
        return -np.sum(probs * np.log2(probs))

    ig_scores = {}
    for col in X_df.columns:
        total_entropy = entropy(y)
        values = X_df[col]
        weighted_entropy = 0
        for v in values.unique():
            subset_y = y[values == v]
            weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)
        ig = total_entropy - weighted_entropy
        if ig >= threshold:
            ig_scores[col] = ig
    return list(ig_scores.keys())

def evaluate_metrics(X, y, test_sizes=[0.1, 0.2]):
    results = {}
    for ts in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        acc = accuracy_score(y_test, y_pred)
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']

        results[f"Split {int((1-ts)*100)}:{int(ts*100)}"] = [acc*100, precision*100, recall*100, f1*100]
    return results

def plot_metrics_comparison(results, title):
    categories = ['Accuracy', 'Precission', 'Recall', 'F1_Score']
    bar_width = 0.35
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['blue', 'green']

    for i, (label, values) in enumerate(results.items()):
        rounded_scores = [round(v) for v in values]
        bars = ax.bar(x + i * bar_width, rounded_scores, width=bar_width, label=label, color=colors[i])
        for bar, score in zip(bars, rounded_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f"{score}%", ha='center', va='bottom', color='white', fontsize=11)

    ax.set_ylabel("Persentase")
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 105)
    ax.legend()
    st.pyplot(fig)

# === DATA ===
data = load_preprocessed_data()
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data["Ulasan"])
X_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(data["Ulasan"])
count_df = pd.DataFrame(X_count.toarray(), columns=count_vectorizer.get_feature_names_out())
y = data["Encoded Label"]

# === SIDEBAR ===
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigasi", ["Home", "Data", "Pengujian", "Report", "Word Cloud", "Prediksi Baru"])

# === HOME ===
if menu == "Home":
    st.title("PENERAPAN METODE RANDOM FOREST DALAM ANALISIS SENTIMEN ULASAN DESTINASI WISATA MADURA DENGAN ENSEMBLE FEATURE SELECTION")
    st.markdown("""
        <h3>Nama: Bagas Pratama Putra</h3>
        <h3>NIM: 200411100184</h3>
    """, unsafe_allow_html=True)

elif menu == "Data":
    st.header("Data")
    
    # Tab untuk data asli dan data preprocessing
    tab1, tab2 = st.tabs(["Data Asli", "Data Preprocessing"])
    
    with tab1:
        st.subheader("Data Asli")
        data_original = load_data()
        st.dataframe(data_original)
        
    with tab2:
        st.subheader("Data Hasil Preprocessing")
        data_preprocessed = load_preprocessed_data()
        st.dataframe(data_preprocessed)

# === WORD CLOUD ===
elif menu == "Word Cloud":
    st.header("Word Cloud")
    col1, col2, col3 = st.columns(3)
    for label, col, cmap in zip(["positif", "negatif", "netral"], [col1, col2, col3], ["Greens", "Reds", "Blues"]):
        with col:
            text = " ".join(data[data["Label"] == label]["Ulasan"].astype(str))
            if text.strip():
                wc = WordCloud(width=300, height=300, background_color="white", colormap=cmap).generate(text)
                st.subheader(label.capitalize())
                st.image(wc.to_array())
            else:
                st.warning(f"Tidak ada data untuk label: {label}")

# === PENGUJIAN ===
elif menu == "Pengujian":
    st.header("Pengujian Random Forest")

    opsi_pengujian = st.radio("Pilih Pengujian:", ["Pengujian 1 (Tanpa Seleksi Fitur)", "Pengujian 2 (Masing-masing Seleksi Fitur)", "Pengujian 3 (Ensemble Seleksi Fitur)", "Pengujian 4 (Masing-masing Seleksi Fitur dengan SMOTE)"])

    if opsi_pengujian == "Pengujian 1 (Tanpa Seleksi Fitur)":
        st.subheader("Tanpa Seleksi Fitur (90:10)")
        
        # --- Split 90:10 ---
        X_train_90, X_test_10, y_train_90, y_test_10 = train_test_split(X_df, y, test_size=0.1, random_state=42, stratify=y)
        model_90 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_90.fit(X_train_90, y_train_90)
        y_pred_10 = model_90.predict(X_test_10)

        report_90 = classification_report(y_test_10, y_pred_10, output_dict=True, zero_division=1)
        acc_90 = accuracy_score(y_test_10, y_pred_10)
        prec_90 = report_90['macro avg']['precision']
        rec_90 = report_90['macro avg']['recall']
        f1_90 = report_90['macro avg']['f1-score']
        
        # Ubah ke format desimal 1 digit
        scores_90 = [f"{acc_90*100:.1f}%", f"{prec_90*100:.1f}%", f"{rec_90*100:.1f}%", f"{f1_90*100:.1f}%"]
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        cm_90 = confusion_matrix(y_test_10, y_pred_10, labels=[-1, 0, 1])

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            bars1 = ax1.bar(categories, [float(x.strip('%')) for x in scores_90], color='blue')
            for bar, score in zip(bars1, scores_90):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, 
                        score, ha='center', va='bottom', color='white', fontsize=11)
            ax1.set_ylabel("Persentase")
            ax1.set_title("Klasifikasi 90:10", fontsize=14)
            ax1.set_ylim(0, 105)
            st.pyplot(fig1)

        with col2:
            fig_cm90, ax_cm90 = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_90, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Negatif", "Netral", "Positif"],
                        yticklabels=["Negatif", "Netral", "Positif"], ax=ax_cm90)
            ax_cm90.set_xlabel("Predicted Label")
            ax_cm90.set_ylabel("True Label")
            ax_cm90.set_title("Confusion Matrix (90:10)")
            st.pyplot(fig_cm90)

        # --- Split 80:20 ---
        st.subheader("Tanpa Seleksi Fitur (80:20)")
        
        X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
        model_80 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_80.fit(X_train_80, y_train_80)
        y_pred_20 = model_80.predict(X_test_20)

        report_80 = classification_report(y_test_20, y_pred_20, output_dict=True, zero_division=1)
        acc_80 = accuracy_score(y_test_20, y_pred_20)
        prec_80 = report_80['macro avg']['precision']
        rec_80 = report_80['macro avg']['recall']
        f1_80 = report_80['macro avg']['f1-score']
        
        # Ubah ke format desimal 1 digit
        scores_80 = [f"{acc_80*100:.1f}%", f"{prec_80*100:.1f}%", f"{rec_80*100:.1f}%", f"{f1_80*100:.1f}%"]

        cm_80 = confusion_matrix(y_test_20, y_pred_20, labels=[-1, 0, 1])

        col3, col4 = st.columns(2)
        with col3:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            bars2 = ax2.bar(categories, [float(x.strip('%')) for x in scores_80], color='green')
            for bar, score in zip(bars2, scores_80):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, 
                        score, ha='center', va='bottom', color='white', fontsize=11)
            ax2.set_ylabel("Persentase")
            ax2.set_title("Klasifikasi 80:20", fontsize=14)
            ax2.set_ylim(0, 105)
            st.pyplot(fig2)

        with col4:
            fig_cm80, ax_cm80 = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_80, annot=True, fmt="d", cmap="Greens",
                        xticklabels=["Negatif", "Netral", "Positif"],
                        yticklabels=["Negatif", "Netral", "Positif"], ax=ax_cm80)
            ax_cm80.set_xlabel("Predicted Label")
            ax_cm80.set_ylabel("True Label")
            ax_cm80.set_title("Confusion Matrix (80:20)")
            st.pyplot(fig_cm80)


    elif opsi_pengujian == "Pengujian 2 (Masing-masing Seleksi Fitur)":
        st.subheader("Pengujian 2: Masing-masing Seleksi Fitur")

        def evaluasi_dua_split(X_selected, y):
            X_train90, X_test10, y_train90, y_test10 = train_test_split(X_selected, y, test_size=0.1, random_state=42)
            X_train80, X_test20, y_train80, y_test20 = train_test_split(X_selected, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Split 90:10
            model.fit(X_train90, y_train90)
            y_pred90 = model.predict(X_test10)
            acc90 = accuracy_score(y_test10, y_pred90)
            cm90 = confusion_matrix(y_test10, y_pred90)

            # Split 80:20
            model.fit(X_train80, y_train80)
            y_pred80 = model.predict(X_test20)
            acc80 = accuracy_score(y_test20, y_pred80)
            cm80 = confusion_matrix(y_test20, y_pred80)

            return acc90, acc80, cm90, cm80

        # ======================================
        # Information Gain
        # ======================================
        st.markdown("### Hasil Information Gain")
        ig_thresholds = [0.001, 0.0025, 0.005]
        ig_acc_90 = []
        ig_acc_80 = []
        ig_conf_matrices = []

        for thresh in ig_thresholds:
            ig_features = info_gain_selection(X_df, y, threshold=thresh)
            acc90, acc80, cm90, cm80 = evaluasi_dua_split(X_df[ig_features], y)
            ig_acc_90.append(acc90)
            ig_acc_80.append(acc80)

            # Simpan confusion matrix terbaik untuk threshold ini
            if acc90 >= acc80:
                ig_conf_matrices.append((thresh, '90:10', cm90))
            else:
                ig_conf_matrices.append((thresh, '80:20', cm80))

        # Grafik Akurasi IG
        fig_ig, ax_ig = plt.subplots(figsize=(8, 5))
        index = np.arange(len(ig_thresholds))
        bar_width = 0.35

        bars_90 = ax_ig.bar(index - bar_width/2, [a*100 for a in ig_acc_90], bar_width, label='90:10 (Biru)', color='blue', edgecolor='black')
        bars_80 = ax_ig.bar(index + bar_width/2, [a*100 for a in ig_acc_80], bar_width, label='80:20 (Hijau)', color='green', edgecolor='black')

        ax_ig.set_title("Akurasi Information Gain berdasarkan Threshold")
        ax_ig.set_xticks(index)
        ax_ig.set_xticklabels([str(t) for t in ig_thresholds])
        ax_ig.set_ylabel("Akurasi (%)")
        ax_ig.set_ylim(0, 100)
        ax_ig.legend()
        for i in range(len(index)):
            ax_ig.text(index[i] - bar_width/2, ig_acc_90[i]*100 + 1, f"{ig_acc_90[i]*100:.2f}%", ha='center')
            ax_ig.text(index[i] + bar_width/2, ig_acc_80[i]*100 + 1, f"{ig_acc_80[i]*100:.2f}%", ha='center')
        st.pyplot(fig_ig)

        # Confusion Matrix terbaik IG
        st.markdown("#### Confusion Matrix Terbaik (Information Gain)")
        cols_ig = st.columns(3)
        for i, (thresh, split, cm) in enumerate(ig_conf_matrices):
            with cols_ig[i]:
                st.markdown(f"**Threshold IG: {thresh}**<br>Split: **{split}**", unsafe_allow_html=True)
                fig_cm, ax = plt.subplots(figsize=(4, 3.5))
                cmap = 'Blues' if split == '90:10' else 'Greens'
                sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                            xticklabels=["Negatif", "Netral", "Positif"],
                            yticklabels=["Negatif", "Netral", "Positif"],
                            cbar=False, ax=ax)
                ax.set_title(None)
                ax.set_xlabel("Predicted", fontsize=9)
                ax.set_ylabel("Actual", fontsize=9)
                st.pyplot(fig_cm)

        # ======================================
        # Chi-Square
        # ======================================
        st.markdown("### Hasil Chi-Square")
        chi_alphas = [0.001, 0.0025, 0.005]
        chi_acc_90 = []
        chi_acc_80 = []
        chi_conf_matrices = []

        for alpha in chi_alphas:
            chi_features = chi_square_selection(count_df, y, alpha=alpha)
            acc90, acc80, cm90, cm80 = evaluasi_dua_split(count_df[chi_features], y)
            chi_acc_90.append(acc90)
            chi_acc_80.append(acc80)

            # Simpan confusion matrix terbaik
            if acc90 >= acc80:
                chi_conf_matrices.append((alpha, '90:10', cm90))
            else:
                chi_conf_matrices.append((alpha, '80:20', cm80))

        # Grafik Akurasi Chi
        fig_chi, ax_chi = plt.subplots(figsize=(8, 5))
        index = np.arange(len(chi_alphas))
        bars_90 = ax_chi.bar(index - bar_width/2, [a*100 for a in chi_acc_90], bar_width, label='90:10 (Biru)', color='blue', edgecolor='black')
        bars_80 = ax_chi.bar(index + bar_width/2, [a*100 for a in chi_acc_80], bar_width, label='80:20 (Hijau)', color='green', edgecolor='black')

        ax_chi.set_title("Akurasi Chi-Square berdasarkan Alpha")
        ax_chi.set_xticks(index)
        ax_chi.set_xticklabels([str(a) for a in chi_alphas])
        ax_chi.set_ylabel("Akurasi (%)")
        ax_chi.set_ylim(0, 100)
        ax_chi.legend()
        for i in range(len(index)):
            ax_chi.text(index[i] - bar_width/2, chi_acc_90[i]*100 + 1, f"{chi_acc_90[i]*100:.2f}%", ha='center')
            ax_chi.text(index[i] + bar_width/2, chi_acc_80[i]*100 + 1, f"{chi_acc_80[i]*100:.2f}%", ha='center')
        st.pyplot(fig_chi)

        # Confusion Matrix terbaik Chi
        st.markdown("#### Confusion Matrix Terbaik (Chi-Square)")
        cols_chi = st.columns(3)
        for i, (alpha, split, cm) in enumerate(chi_conf_matrices):
            with cols_chi[i]:
                st.markdown(f"**Alpha: {alpha}**<br>Split: **{split}**", unsafe_allow_html=True)
                fig_cm, ax = plt.subplots(figsize=(4, 3.5))
                cmap = 'Blues' if split == '90:10' else 'Greens'
                sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                            xticklabels=["Negatif", "Netral", "Positif"],
                            yticklabels=["Negatif", "Netral", "Positif"],
                            cbar=False, ax=ax)
                ax.set_title(None)
                ax.set_xlabel("Predicted", fontsize=9)
                ax.set_ylabel("Actual", fontsize=9)
                st.pyplot(fig_cm)

    elif opsi_pengujian == "Pengujian 3 (Ensemble Seleksi Fitur)":
        st.subheader("Pengujian 3: Ensemble Feature Selection")

        # --- Langkah 1: Seleksi fitur ---
        ig_features = info_gain_selection(X_df, y)          # IG dari TF-IDF
        chi_features = chi_square_selection(count_df, y)    # Chi-Square dari CountVectorizer

        # Ensemble features
        intersection_features = list(set(ig_features) & set(chi_features))
        union_features = list(set(ig_features) | set(chi_features))

        # --- Langkah 2: Gabungkan fitur dari dua sumber sesuai hasil ensemble ---
        def ambil_fitur_terpilih(feature_list):
            # Ambil kolom dari X_df dan count_df jika nama fitur cocok
            tfidf_part = X_df[[feat for feat in feature_list if feat in X_df.columns]]
            count_part = count_df[[feat for feat in feature_list if feat in count_df.columns]]
            return pd.concat([tfidf_part, count_part], axis=1)

        X_intersection = ambil_fitur_terpilih(intersection_features)
        X_union = ambil_fitur_terpilih(union_features)

        # --- Langkah 3: Fungsi evaluasi model ---
        def evaluasi_model(X, y, judul):
            try:
                # Split 90:10 dan 80:20
                X_train90, X_test10, y_train90, y_test10 = train_test_split(X, y, test_size=0.1, random_state=42)
                X_train80, X_test20, y_train80, y_test20 = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)

                # Train & predict
                model.fit(X_train90, y_train90)
                y_pred10 = model.predict(X_test10)
                report90 = classification_report(y_test10, y_pred10, output_dict=True)

                model.fit(X_train80, y_train80)
                y_pred20 = model.predict(X_test20)
                report80 = classification_report(y_test20, y_pred20, output_dict=True)

                # Layout hasil
                col1, col2 = st.columns(2)
                with col1:
                    # Grafik metrik
                    fig, ax = plt.subplots(figsize=(7, 5))
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    values90 = [
                        round(report90['accuracy'] * 100, 1),
                        round(report90['macro avg']['precision'] * 100, 1),
                        round(report90['macro avg']['recall'] * 100, 1),
                        round(report90['macro avg']['f1-score'] * 100, 1)
                    ]
                    values80 = [
                        round(report80['accuracy'] * 100, 1),
                        round(report80['macro avg']['precision'] * 100, 1),
                        round(report80['macro avg']['recall'] * 100, 1),
                        round(report80['macro avg']['f1-score'] * 100, 1)
                    ]
                    x = np.arange(len(metrics))
                    width = 0.35
                    bars90 = ax.bar(x - width/2, values90, width, label='90:10', color='blue', edgecolor='black')
                    bars80 = ax.bar(x + width/2, values80, width, label='80:20', color='green', edgecolor='black')

                    ax.set_title('Perbandingan Klasifikasi pada Pembagian Data 90:10 dan 80:20',
                                pad=20, fontsize=14, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(metrics, fontsize=12)
                    ax.set_ylabel('Percentage (%)', fontsize=12)
                    ax.set_ylim(0, 110)
                    ax.grid(axis='y', linestyle='--', alpha=0.3)

                    for bars in [bars90, bars80]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2.,
                                    height + 1,
                                    f'{height:.1f}%',
                                    ha='center',
                                    va='bottom',
                                    color='black',
                                    fontsize=11)
                    ax.legend(loc='upper right', framealpha=1, shadow=True)
                    st.pyplot(fig)

                with col2:
                    # Confusion matrix dari split terbaik
                    best_acc = max(report90['accuracy'], report80['accuracy'])
                    if report90['accuracy'] >= report80['accuracy']:
                        cm = confusion_matrix(y_test10, y_pred10, labels=[-1, 0, 1])
                        cm_color = 'Blues'
                        split_label = '90:10'
                    else:
                        cm = confusion_matrix(y_test20, y_pred20, labels=[-1, 0, 1])
                        cm_color = 'Greens'
                        split_label = '80:20'

                    fig_cm, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap=cm_color,
                                xticklabels=["Negatif", "Netral", "Positif"],
                                yticklabels=["Negatif", "Netral", "Positif"], ax=ax)
                    ax.set_title(f'Confusion Matrix ({split_label})\nAkurasi: {best_acc*100:.1f}%', 
                                fontsize=12)
                    st.pyplot(fig_cm)

            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")

        # --- Langkah 4: Evaluasi intersection dan union ---
        st.markdown("### Ensemble Intersection (IG ∩ Chi-Square)")
        evaluasi_model(X_intersection, y, "Intersection")

        st.markdown("### Ensemble Union (IG ∪ Chi-Square)")
        evaluasi_model(X_union, y, "Union")

    elif opsi_pengujian == "Pengujian 4 (Masing-masing Seleksi Fitur dengan SMOTE)":
        st.subheader("Pengujian 4: Masing-masing Seleksi Fitur dengan SMOTE")

        def evaluasi_smote(X_selected, y):
            acc_90_list, acc_80_list, best_cm_info = [], [], []

            for X_feat in X_selected:
                # Split data
                X_train90, X_test10, y_train90, y_test10 = train_test_split(X_feat, y, test_size=0.1, random_state=42)
                X_train80, X_test20, y_train80, y_test20 = train_test_split(X_feat, y, test_size=0.2, random_state=42)

                # SMOTE
                sm = SMOTE(random_state=42)
                X_train90_sm, y_train90_sm = sm.fit_resample(X_train90, y_train90)
                X_train80_sm, y_train80_sm = sm.fit_resample(X_train80, y_train80)

                # Model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Evaluasi 90:10
                model.fit(X_train90_sm, y_train90_sm)
                y_pred10 = model.predict(X_test10)
                acc90 = accuracy_score(y_test10, y_pred10)
                
                # Evaluasi 80:20
                model.fit(X_train80_sm, y_train80_sm)
                y_pred20 = model.predict(X_test20)
                acc80 = accuracy_score(y_test20, y_pred20)

                # Simpan hasil
                acc_90_list.append(acc90)
                acc_80_list.append(acc80)

                if acc90 >= acc80:
                    best_cm_info.append((y_test10, y_pred10, '90:10'))
                else:
                    best_cm_info.append((y_test20, y_pred20, '80:20'))

            return acc_90_list, acc_80_list, best_cm_info

        # ====================
        # 1. Information Gain
        # ====================
        st.markdown("### Hasil Information Gain + SMOTE")
        ig_thresholds = [0.001, 0.0025, 0.005]
        ig_features = [info_gain_selection(X_df, y, threshold=t) for t in ig_thresholds]
        X_ig_selected = [X_df[feat] for feat in ig_features]

        acc90_ig, acc80_ig, best_cm_ig = evaluasi_smote(X_ig_selected, y)

        # Visualisasi IG
        x = np.arange(len(ig_thresholds))
        width = 0.35
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars1 = ax1.bar(x - width/2, [a * 100 for a in acc90_ig], width, label='90:10', color='blue', edgecolor='black')
        bars2 = ax1.bar(x + width/2, [a * 100 for a in acc80_ig], width, label='80:20', color='green', edgecolor='black')

        ax1.set_xlabel('Threshold IG')
        ax1.set_ylabel('Akurasi (%)')
        ax1.set_title('Akurasi Random Forest + SMOTE berdasarkan Threshold IG')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(t) for t in ig_thresholds])
        ax1.set_ylim(0, 110)
        ax1.legend()

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(bars1)
        autolabel(bars2)
        st.pyplot(fig1)

        # Confusion matrix IG
        st.markdown("#### Confusion Matrix IG")
        cols = st.columns(len(ig_thresholds))
        for i, col in enumerate(cols):
            with col:
                y_true, y_pred, split = best_cm_ig[i]
                cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues' if split == '90:10' else 'Greens',
                            xticklabels=["Negatif", "Netral", "Positif"],
                            yticklabels=["Negatif", "Netral", "Positif"],
                            cbar=False, ax=ax_cm)
                ax_cm.set_title(f'Alpha {ig_thresholds[i]} ({split})')
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

        # ====================
        # 2. Chi-Square
        # ====================
        st.markdown("### Hasil Chi-Square + SMOTE")
        chi_alphas = [0.001, 0.0025, 0.005]
        chi_features = [chi_square_selection(count_df, y, alpha=a) for a in chi_alphas]
        X_chi_selected = [count_df[feat] for feat in chi_features]

        acc90_chi, acc80_chi, best_cm_chi = evaluasi_smote(X_chi_selected, y)

        # Visualisasi Chi-Square
        x = np.arange(len(chi_alphas))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bars1 = ax2.bar(x - width/2, [a * 100 for a in acc90_chi], width, label='90:10', color='blue', edgecolor='black')
        bars2 = ax2.bar(x + width/2, [a * 100 for a in acc80_chi], width, label='80:20', color='green', edgecolor='black')

        ax2.set_xlabel('Alpha Chi-Square')
        ax2.set_ylabel('Akurasi (%)')
        ax2.set_title('Akurasi Random Forest + SMOTE berdasarkan Alpha Chi-Square')
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(a) for a in chi_alphas])
        ax2.set_ylim(0, 110)
        ax2.legend()

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.annotate(f'{height:.2f}%',  # Format sebagai persentase
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # Offset teks
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)

        autolabel(bars1)
        autolabel(bars2)
        st.pyplot(fig2)

        # Confusion matrix Chi-Square
        st.markdown("#### Confusion Matrix Chi-Square")
        cols = st.columns(len(chi_alphas))
        for i, col in enumerate(cols):
            with col:
                y_true, y_pred, split = best_cm_chi[i]
                cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues' if split == '90:10' else 'Greens',
                            xticklabels=["Negatif", "Netral", "Positif"],
                            yticklabels=["Negatif", "Netral", "Positif"],
                            cbar=False, ax=ax_cm)
                ax_cm.set_title(f'Alpha {chi_alphas[i]} ({split})')
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

        # ==================================================
        # Information Gain TANPA kelas netral (Binary Class)
        # ==================================================
        st.markdown("### Hasil Information Gain + SMOTE (2 Kelas)")

        # Filter hanya label -1 dan 1
        mask_bin = y.isin([-1, 1])
        y_bin = y[mask_bin]
        X_ig_bin = X_df[mask_bin]  # Gunakan TF-IDF sebagai dasar IG

        ig_thresholds = [0.001, 0.0025, 0.005]
        ig_features_bin = [info_gain_selection(X_ig_bin, y_bin, threshold=t) for t in ig_thresholds]
        X_ig_selected_bin = [X_ig_bin[feat] for feat in ig_features_bin]

        acc90_ig_bin, acc80_ig_bin, best_cm_ig_bin = evaluasi_smote(X_ig_selected_bin, y_bin)

        # Visualisasi grafik
        x = np.arange(len(ig_thresholds))
        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width/2, [a * 100 for a in acc90_ig_bin], width, label='90:10', color='blue')
        bars2 = ax.bar(x + width/2, [a * 100 for a in acc80_ig_bin], width, label='80:20', color='green')

        ax.set_xlabel('Threshold IG')
        ax.set_ylabel('Akurasi (%)')
        ax.set_title('Akurasi Random Forest + SMOTE berdasarkan IG Threshold (2 Kelas)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ig_thresholds])
        ax.set_ylim(0, 110)
        ax.legend()
        autolabel(bars1)
        autolabel(bars2)
        st.pyplot(fig)

        # Confusion matrix IG Binary
        st.markdown("#### Confusion Matrix Information Gain (2 Kelas)")
        cols = st.columns(len(ig_thresholds))
        for i, col in enumerate(cols):
            with col:
                y_true, y_pred, split = best_cm_ig_bin[i]
                cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues' if split == '90:10' else 'Greens',
                            xticklabels=["Negatif", "Positif"],
                            yticklabels=["Negatif", "Positif"],
                            cbar=False, ax=ax_cm)
                ax_cm.set_title(f'Threshold {ig_thresholds[i]} ({split})')
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
        
        # ====================
        # Pengujian 4 Dua Kelas (Chi-Square + SMOTE)
        # ====================
        st.markdown("### Hasil Chi-Square + SMOTE (2 Kelas)")

        # Filter data untuk hanya dua kelas: -1 dan 1
        mask = y.isin([-1, 1])
        y_bin = y[mask]
        count_df_bin = count_df.loc[mask]

        # Lakukan seleksi fitur untuk masing-masing alpha
        chi_features_bin = [chi_square_selection(count_df_bin, y_bin, alpha=a) for a in chi_alphas]
        X_chi_bin_selected = [count_df_bin[feat] for feat in chi_features_bin]

        # Evaluasi dengan SMOTE
        acc90_chi_bin, acc80_chi_bin, best_cm_chi_bin = evaluasi_smote(X_chi_bin_selected, y_bin)

        # Visualisasi akurasi Chi-Square Tanpa Netral
        x = np.arange(len(chi_alphas))
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bars1_bin = ax3.bar(x - width/2, [a * 100 for a in acc90_chi_bin], width, label='90:10', color='blue')
        bars2_bin = ax3.bar(x + width/2, [a * 100 for a in acc80_chi_bin], width, label='80:20', color='green')

        ax3.set_xlabel('Alpha Chi-Square')
        ax3.set_ylabel('Akurasi (%)')
        ax3.set_title('Akurasi RF + SMOTE 2 Kelas berdasarkan Alpha Chi-Square')
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(a) for a in chi_alphas])
        ax3.set_ylim(0, 110)
        ax3.legend()

        autolabel(bars1_bin)
        autolabel(bars2_bin)
        st.pyplot(fig3)

        # Confusion matrix Chi-Square Tanpa Netral
        st.markdown("#### Confusion Matrix Chi-Square (2 Kelas)")
        cols_bin = st.columns(len(chi_alphas))
        for i, col in enumerate(cols_bin):
            with col:
                y_true, y_pred, split = best_cm_chi_bin[i]
                cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
                fig_cm_bin, ax_cm_bin = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap='Blues' if split == '90:10' else 'Greens',
                            xticklabels=["Negatif", "Positif"],
                            yticklabels=["Negatif", "Positif"],
                            cbar=False, ax=ax_cm_bin)
                ax_cm_bin.set_title(f'Alpha {chi_alphas[i]} ({split})')
                ax_cm_bin.set_xlabel("Predicted")
                ax_cm_bin.set_ylabel("Actual")
                st.pyplot(fig_cm_bin)

elif menu == "Report":
    st.subheader("Perbandingan Akurasi Semua Pengujian (1, 2, dan 3)")

    # Data akurasi (ganti dengan hasil aktual jika sudah tersedia)
    data = {
        "Pengujian": [
            "P1 (90:10)", "P1 (80:20)",
            "P2 IG 0.001 (90:10)", "P2 IG 0.0025 (90:10)", "P2 IG 0.005 (90:10)",
            "P2 IG 0.001 (80:20)", "P2 IG 0.0025 (80:20)", "P2 IG 0.005 (80:20)",
            "P2 Chi2 0.001 (90:10)", "P2 Chi2 0.0025 (90:10)", "P2 Chi2 0.005 (90:10)",
            "P2 Chi2 0.001 (80:20)", "P2 Chi2 0.0025 (80:20)", "P2 Chi2 0.005 (80:20)",
            "P3 Union (90:10)", "P3 Union (80:20)",
            "P3 Intersection (90:10)", "P3 Intersection (80:20)"
        ],
        "Akurasi": [
            0.753, 0.773,
            0.799, 0.7835, 0.7577,
            0.7752, 0.77, 0.7674,
            0.7371, 0.7371, 0.732,
            0.7261, 0.7235, 0.7106,
            0.784, 0.778,
            0.696, 0.682
        ],
        "Split": [
            "90:10", "80:20",
            "90:10", "90:10", "90:10",
            "80:20", "80:20", "80:20",
            "90:10", "90:10", "90:10",
            "80:20", "80:20", "80:20",
            "90:10", "80:20",
            "90:10", "80:20"
        ]
    }

    df = pd.DataFrame(data)

    # Warna berdasarkan split
    colors = df["Split"].map({"90:10": "blue", "80:20": "green"})

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(df["Pengujian"], df["Akurasi"], color=colors)

    ax.set_xlabel("Pengujian")
    ax.set_ylabel("Akurasi")
    ax.set_title("Perbandingan Hasil Akurasi Pengujian 1,2 dan 3")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(df["Pengujian"], rotation=45, ha='right')

    # Tambahkan nilai persentase di atas batang
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{height*100:.1f}%", ha='center', va='bottom', fontsize=9)

    # Legenda
    legend_elements = [
        Patch(facecolor='blue', label='Split 90:10'),
        Patch(facecolor='green', label='Split 80:20')
    ]
    ax.legend(handles=legend_elements)

    st.pyplot(fig)

    st.subheader("Perbandingan Hasil Akurasi Pengujian 4 Seleksi Fitur + SMOTE (2 Kelas vs 3 Kelas)")

    # Data Gabungan
    data = {
        "Pengujian": [
            "IG 0.001 (90:10)", "IG 0.0025 (90:10)", "IG 0.005 (90:10)",
            "IG 0.001 (80:20)", "IG 0.0025 (80:20)", "IG 0.005 (80:20)",
            "Chi2 0.001 (90:10)", "Chi2 0.0025 (90:10)", "Chi2 0.005 (90:10)",
            "Chi2 0.001 (80:20)", "Chi2 0.0025 (80:20)", "Chi2 0.005 (80:20)"
        ] * 2,
        "Split": (["90:10"] * 3 + ["80:20"] * 3 + ["90:10"] * 3 + ["80:20"] * 3) * 2,
        "Kelas": ["3 Kelas"] * 12 + ["2 Kelas"] * 12,
        "Akurasi": [
            # 3 Kelas
            0.7526, 0.7526, 0.6959,
            0.7442, 0.7416, 0.6899,
            0.5619, 0.5773, 0.5979,
            0.5814, 0.5736, 0.5995,
            # 2 Kelas
            0.7663, 0.7717, 0.7717,
            0.7984, 0.7847, 0.8147,
            0.7011, 0.6957, 0.7283,
            0.7193, 0.7248, 0.7548
        ]
    }

    df = pd.DataFrame(data)

    # Label untuk sumbu X
    df["Label"] = df["Pengujian"] + " (" + df["Split"] + ") - " + df["Kelas"]

    # Warna berdasarkan Split dan Kelas
    def get_color(row):
        if row["Kelas"] == "2 Kelas":
            return "skyblue" if row["Split"] == "90:10" else "lightgreen"
        else:
            return "blue" if row["Split"] == "90:10" else "green"

    df["Color"] = df.apply(get_color, axis=1)

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(df["Label"], df["Akurasi"], color=df["Color"])

    ax.set_xlabel("Pengujian")
    ax.set_ylabel("Akurasi")
    ax.set_title("Perbandingan Akurasi SMOTE + Seleksi Fitur untuk 2 Kelas dan 3 Kelas")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(df["Label"], rotation=45, ha='right')

    # Tampilkan nilai akurasi di atas batang
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f"{height*100:.1f}%", ha='center', va='bottom', fontsize=9)

    # Legenda
    legend_elements = [
        Patch(facecolor='blue', label='3 Kelas - Split 90:10'),
        Patch(facecolor='green', label='3 Kelas - Split 80:20'),
        Patch(facecolor='skyblue', label='2 Kelas - Split 90:10'),
        Patch(facecolor='lightgreen', label='2 Kelas - Split 80:20')
    ]
    ax.legend(handles=legend_elements)

    st.pyplot(fig)

# === PREDIKSI BARU ===
elif menu == "Prediksi Baru":
    st.header("Prediksi Ulasan Baru")
    user_input = st.text_area("Masukkan teks ulasan:")
    
    if st.button("Prediksi"):
        # Preprocessing teks input
        input_clean = preprocess_text(user_input)
        
        # Vectorize teks (gunakan vectorizer yang sama dengan saat training)
        input_vec = vectorizer.transform([input_clean])
        
        # Buat DataFrame dengan semua fitur (tanpa seleksi)
        input_df = pd.DataFrame(input_vec.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Gunakan model dari Pengujian 1 (split 90:10)
        model_80 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_80.fit(X_df, y)  
        
        # Prediksi
        pred = model_80.predict(input_df)[0]
        proba = model_80.predict_proba(input_df)[0]
        
        # Mapping label dan warna
        label_map = {-1: "Negatif", 0: "Netral", 1: "Positif"}
        color_map = {
            -1: "red",  # Merah muda untuk Negatif
            0: "blue",   # Biru muda untuk Netral
            1: "green"     # Hijau muda untuk Positif
        }
        
        # Tampilkan hasil dengan warna background
        st.markdown(
            f'<div style="background-color: {color_map[pred]}; padding: 10px; border-radius: 5px; margin-bottom: 20px;">'
            f'<strong>Hasil Prediksi:</strong> {label_map[pred]}<br>'
            f'<strong>Confidence Score:</strong> {max(proba)*100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Debug info dengan expander agar lebih rapi
        with st.expander("Detail Preprocessing"):
            st.write(f"**Teks setelah preprocessing:**\n{input_clean}")
            st.write(f"**Kata-kata penting yang terdeteksi:**\n{input_df.loc[:, (input_df != 0).any(axis=0)].columns.tolist()}")
