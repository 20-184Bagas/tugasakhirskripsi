import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import chi2
from scipy.stats import chi2 as chi2_table
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import nltk
nltk.download('stopwords')

# === NORMALISASI KAMUS ===
@st.cache_data
def load_normalization_dict():
    url = "https://raw.githubusercontent.com/20-184Bagas/tugasakhirskripsi/refs/heads/main/Kamus_Normalization-_2_%20(1).csv"
    df_norm = pd.read_csv(url)
    return dict(zip(df_norm['Tidak Baku'], df_norm['Baku']))

normalization_dict = load_normalization_dict()

# === FUNGSI DATA & PREPROCESSING ===
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/20-184Bagas/tugasakhirskripsi/refs/heads/main/data-ulasan-wisata-madura.csv")
    df["Label"] = df["Label"].str.lower()
    label_mapping = {"positif": 1, "negatif": -1, "netral": 0}
    df["Encoded Label"] = df["Label"].map(label_mapping)
    df["Ulasan"] = df["Ulasan"].astype(str).apply(preprocess_text)
    return df

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

# === SELEKSI FITUR ===
def chi_square_selection(X, y, alpha=0.0025):
    scores, _ = chi2(X, y)
    df_chi = pd.DataFrame({"Fitur": X.columns, "Chi2": scores})
    chi_crit = chi2_table.ppf(1 - alpha, df=len(set(y)) - 1)
    selected = df_chi[df_chi["Chi2"] >= chi_crit]["Fitur"].tolist()
    return selected

def info_gain_selection(X_df, y, threshold=0.0025):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
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
data = load_data()
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data["Ulasan"])
X_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
y = data["Encoded Label"]

# === SIDEBAR ===
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigasi", ["Home", "Data", "Pengujian", "Word Cloud", "Prediksi Baru"])

# === HOME ===
if menu == "Home":
    st.title("PENERAPAN METODE RANDOM FOREST DALAM ANALISIS SENTIMEN ULASAN DESTINASI WISATA MADURA DENGAN ENSEMBLE FEATURE SELECTION")
     st.markdown("""
    **Nama:** Bagas Pratama Putra  
    **NIM:** 200411100184
    """)

# === DATA ===
elif menu == "Data":
    st.header("📄 Data")
    st.dataframe(data)

# === WORD CLOUD ===
elif menu == "Word Cloud":
    st.header("☁️ Word Cloud")
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
    st.header("🧪 Pengujian Random Forest")

    opsi_pengujian = st.radio("Pilih Pengujian", ["Pengujian 1", "Pengujian 2", "Pengujian 3"])

    if opsi_pengujian == "Pengujian 1":
        st.subheader("📌 Pengujian 1: Tanpa Seleksi Fitur (90:10)")
    
        # --- Split 90:10 ---
        X_train_90, X_test_10, y_train_90, y_test_10 = train_test_split(X_df, y, test_size=0.1, random_state=42)
        model_90 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_90.fit(X_train_90, y_train_90)
        y_pred_10 = model_90.predict(X_test_10)

        report_90 = classification_report(y_test_10, y_pred_10, output_dict=True, zero_division=1)
        acc_90 = accuracy_score(y_test_10, y_pred_10)
        prec_90 = report_90['macro avg']['precision']
        rec_90 = report_90['macro avg']['recall']
        f1_90 = report_90['macro avg']['f1-score']
        scores_90 = [round(acc_90 * 100), round(prec_90 * 100), round(rec_90 * 100), round(f1_90 * 100)]
        categories = ['Accuracy', 'Precission', 'Recall', 'F1_Score']

        cm_90 = confusion_matrix(y_test_10, y_pred_10, labels=[-1, 0, 1])

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            bars1 = ax1.bar(categories, scores_90, color='blue')
            for bar, score in zip(bars1, scores_90):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f"{score}%", ha='center', va='bottom', color='white', fontsize=11)
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
        st.subheader("📌 Pengujian 1: Tanpa Seleksi Fitur (80:20)")
    
        X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X_df, y, test_size=0.2, random_state=42)
        model_80 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_80.fit(X_train_80, y_train_80)
        y_pred_20 = model_80.predict(X_test_20)

        report_80 = classification_report(y_test_20, y_pred_20, output_dict=True, zero_division=1)
        acc_80 = accuracy_score(y_test_20, y_pred_20)
        prec_80 = report_80['macro avg']['precision']
        rec_80 = report_80['macro avg']['recall']
        f1_80 = report_80['macro avg']['f1-score']
        scores_80 = [round(acc_80 * 100), round(prec_80 * 100), round(rec_80 * 100), round(f1_80 * 100)]

        cm_80 = confusion_matrix(y_test_20, y_pred_20, labels=[-1, 0, 1])

        col3, col4 = st.columns(2)
        with col3:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            bars2 = ax2.bar(categories, scores_80, color='green')
            for bar, score in zip(bars2, scores_80):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, f"{score}%", ha='center', va='bottom', color='white', fontsize=11)
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
        
    elif opsi_pengujian == "Pengujian 2":
        st.subheader("📌 Pengujian 2: Evaluasi Threshold Seleksi Fitur")
    
        def evaluasi_dan_tampil(X_selected, y, metode_name, threshold):
        # Evaluasi kedua split
            X_train90, X_test10, y_train90, y_test10 = train_test_split(X_selected, y, test_size=0.1, random_state=42)
            X_train80, X_test20, y_train80, y_test20 = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
            # Hitung metrik untuk 90:10
            model.fit(X_train90, y_train90)
            y_pred10 = model.predict(X_test10)
            report90 = classification_report(y_test10, y_pred10, output_dict=True)
        
            # Hitung metrik untuk 80:20
            model.fit(X_train80, y_train80)
            y_pred20 = model.predict(X_test20)
            report80 = classification_report(y_test20, y_pred20, output_dict=True)
        
            # Tentukan split terbaik
            acc90 = report90['accuracy']
            acc80 = report80['accuracy']
            best_split = '90:10' if acc90 >= acc80 else '80:20'
            cm_data = (y_test10, y_pred10) if best_split == '90:10' else (y_test20, y_pred20)
            cm_color = 'Blues' if best_split == '90:10' else 'Greens'
        
            # Buat layout kolom
            col1, col2 = st.columns(2)
        
            with col1:
                # Grafik perbandingan metrik
                metrics = ['accuracy', 'precision', 'recall', 'f1-score']
                labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
                fig_metrics, ax = plt.subplots(figsize=(7, 5))
            
                # Data untuk plotting
                values90 = [report90[m] if m == 'accuracy' else report90['macro avg'][m] for m in metrics]
                values80 = [report80[m] if m == 'accuracy' else report80['macro avg'][m] for m in metrics]
            
                # Konversi ke persentase
                values90_pct = [v * 100 for v in values90]
                values80_pct = [v * 100 for v in values80]
            
                # Plot bars
                bar_width = 0.35
                index = np.arange(len(metrics))
            
                bars90 = ax.bar(index - bar_width/2, values90_pct, bar_width, 
                          label='90:10', color='blue', edgecolor='black')
                bars80 = ax.bar(index + bar_width/2, values80_pct, bar_width, 
                          label='80:20', color='green', edgecolor='black')
            
                # Formatting
                ax.set_title('Perbandingan Klasifikasi pada Pembagian Data 90:10 dan 80:20', 
                            pad=20, fontsize=12, fontweight='bold')
                ax.set_xticks(index)
                ax.set_xticklabels(labels, fontsize=10)
                ax.set_ylabel('Persentase (%)', fontsize=10)
                ax.set_ylim(0, 110)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.legend(fontsize=9, framealpha=0.9)
            
                # Tambah nilai di atas bar
                for bars in [bars90, bars80]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}%',
                                ha='center', va='bottom', fontsize=9)
            
                plt.tight_layout()
                st.pyplot(fig_metrics)
        
            with col2:
                # Confusion matrix
                cm = confusion_matrix(cm_data[0], cm_data[1], labels=[-1, 0, 1])
                fig_cm, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap=cm_color,
                        xticklabels=["Negatif", "Netral", "Positif"],
                        yticklabels=["Negatif", "Netral", "Positif"],
                        ax=ax, cbar=False)
                ax.set_title(f'Confusion Matrix ({best_split})', pad=15, fontsize=12)
                ax.set_xlabel("Predicted", fontsize=10)
                ax.set_ylabel("Actual", fontsize=10)
                st.pyplot(fig_cm)
    # =================================================
    # 1. Information Gain
    # =================================================
        st.markdown("## 🔷 Hasil Information Gain")
        ig_thresholds = [0.0025, 0.005, 0.001]
        for thresh in ig_thresholds:
            st.markdown(f"### 🔹 IG Threshold = {thresh}")
            ig_features = info_gain_selection(X_df, y, threshold=thresh)
            evaluasi_dan_tampil(X_df[ig_features], y, "Information Gain", thresh)
    
    # ================================================
    # 2. Chi-Square
    # ================================================
        st.markdown("## 🔶 Hasil Chi-Square")
        chi_alphas = [0.0025, 0.005, 0.001]
        for alpha in chi_alphas:
            st.markdown(f"### 🔸 Chi-Square Alpha = {alpha}")
            chi_features = chi_square_selection(X_df, y, alpha=alpha)
            evaluasi_dan_tampil(X_df[chi_features], y, "Chi-Square", alpha)

    elif opsi_pengujian == "Pengujian 3":
        st.subheader("📌 Pengujian 3: Ensemble Feature Selection")

        # Hitung fitur seleksi
        ig_features = info_gain_selection(X_df, y)
        chi_features = chi_square_selection(X_df, y)

        intersection_features = list(set(ig_features) & set(chi_features))
        union_features = list(set(ig_features) | set(chi_features))

        def evaluasi_model(X, y, judul):
            try:
                # Evaluasi kedua split
                X_train90, X_test10, y_train90, y_test10 = train_test_split(X, y, test_size=0.1, random_state=42)
                X_train80, X_test20, y_train80, y_test20 = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Hitung metrik untuk 90:10
                model.fit(X_train90, y_train90)
                y_pred10 = model.predict(X_test10)
                report90 = classification_report(y_test10, y_pred10, output_dict=True)
                
                # Hitung metrik untuk 80:20
                model.fit(X_train80, y_train80)
                y_pred20 = model.predict(X_test20)
                report80 = classification_report(y_test20, y_pred20, output_dict=True)
                
                # Buat layout kolom
                col1, col2 = st.columns(2)
                
                with col1:
                    # Grafik perbandingan
                    fig, ax = plt.subplots(figsize=(7, 5))
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    
                    # Konversi ke persentase dengan 1 desimal
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
                    
                    # Plot bars
                    x = np.arange(len(metrics))
                    width = 0.35
                    
                    bars90 = ax.bar(x - width/2, values90, width, label='90:10', color='blue', edgecolor='black')
                    bars80 = ax.bar(x + width/2, values80, width, label='80:20', color='green', edgecolor='black')
                    
                    # Formatting
                    ax.set_title('Perbandingan Klasifikasi pada Pembagian Data 90:10 dan 80:20', 
                                pad=20, fontsize=14, fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(metrics, fontsize=12)
                    ax.set_ylabel('Percentage (%)', fontsize=12)
                    ax.set_ylim(0, 110)
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    # Nilai di atas bar (format XX.X% dengan font hitam)
                    for bars in [bars90, bars80]:
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., 
                                    height + 1,  # Posisi 1% di atas bar
                                    f'{height:.1f}%', 
                                    ha='center', 
                                    va='bottom',
                                    color='black',  # Warna font hitam
                                    fontsize=11,
                                    fontweight='normal')
                    
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

        # --- Bagian 1: Ensemble Intersection ---
        st.markdown("### 🔸 Ensemble Intersection (IG ∩ Chi-Square)")
        evaluasi_model(X_df[intersection_features], y, "Intersection")

        # --- Bagian 2: Ensemble Union ---
        st.markdown("### 🔸 Ensemble Union (IG ∪ Chi-Square)")
        evaluasi_model(X_df[union_features], y, "Union")

# === PREDIKSI BARU ===
elif menu == "Prediksi Baru":
    st.header("🔮 Prediksi Ulasan Baru")
    user_input = st.text_area("Masukkan teks ulasan:")
    
    if st.button("Prediksi"):
        # Preprocessing teks input
        input_clean = preprocess_text(user_input)
        
        # Vectorize teks (gunakan vectorizer yang sama dengan saat training)
        input_vec = vectorizer.transform([input_clean])
        
        # Buat DataFrame dengan semua fitur (tanpa seleksi)
        input_df = pd.DataFrame(input_vec.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Gunakan model dari Pengujian 1 (split 90:10)
        model_90 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_90.fit(X_df, y)  # Asumsi X_df dan y sudah didefinisikan sebelumnya
        
        # Prediksi
        pred = model_90.predict(input_df)[0]
        proba = model_90.predict_proba(input_df)[0]
        
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
        with st.expander("🔍 Detail Preprocessing"):
            st.write(f"**Teks setelah preprocessing:**\n{input_clean}")
            st.write(f"**Kata-kata penting yang terdeteksi:**\n{input_df.loc[:, (input_df != 0).any(axis=0)].columns.tolist()}")
