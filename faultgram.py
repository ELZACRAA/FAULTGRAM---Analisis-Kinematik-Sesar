"""
STEREONET-DIGITAL APP
Klasifikasi Sesar Rickard 1971 & Analisis Kekar Konjugasi
===========================================================
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Mengamankan backend matplotlib agar tidak crash di Streamlit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
    import mplstereonet
    HAS_MPLSTEREONET = True
except ImportError:
    HAS_MPLSTEREONET = False

st.set_page_config(page_title="FAULTGRAM", layout="wide", initial_sidebar_state="expanded")


# =====================================================================
# FUNGSI GEOMETRI & KONVERSI DASAR (VERSI ASLI)
# =====================================================================

def strike_dip_to_pole(strike, dip):
    strike_r = np.radians(strike)
    dip_dir_r = strike_r + np.pi / 2
    plunge = np.radians(90 - dip)
    x = np.cos(plunge) * np.cos(dip_dir_r)
    y = np.cos(plunge) * np.sin(dip_dir_r)
    z = np.sin(plunge)
    return np.array([x, y, z])


def vector_to_trend_plunge(vec):
    x, y, z = vec
    if z < 0:
        x, y, z = -x, -y, -z
    plunge = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    trend = np.degrees(np.arctan2(y, x)) % 360
    return trend, plunge


def normal_to_plane(vec):
    trend, plunge = vector_to_trend_plunge(vec)
    strike = (trend + 90) % 360
    dip = 90 - plunge
    return strike, dip

def line_to_vector(trend, plunge):
    trend_r = np.radians(trend)
    plunge_r = np.radians(plunge)
    return np.array([np.cos(plunge_r) * np.cos(trend_r),np.cos(plunge_r) * np.sin(trend_r),np.sin(plunge_r)])

def angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


def hitung_arah_mata_angin(azimuth):
    """Mengonversi derajat azimuth menjadi teks arah mata angin kualitatif."""
    azimuth = azimuth % 360
    arah = [
        "Utara (N)", "Timur Laut (NE)", "Timur (E)", "Tenggara (SE)",
        "Selatan (S)", "Barat Daya (SW)", "Barat (W)", "Barat Laut (NW)"
    ]
    indeks = int((azimuth + 22.5) / 45) % 8
    return arah[indeks]


def klasifikasi_sesar_rickard_1971(fault_strike, fault_dip, pitch, s1_trend, s1_plunge, s3_plunge, s2_plunge):
    """
    Menentukan nama sesar berdasarkan 22 penamaan resmi klasifikasi
    Rickard (1971), memakai kriteria yang KONSISTEN untuk setiap band pitch.
    """
 
    # =================================================================
    # 1. SENSE OF SHEAR (MENGANAN / MENGIRI) - dihitung sebagai AXIS
    #    strike dan trend sigma1 tidak punya arah (mod 180), sehingga
    #    dibandingkan mod 180, bukan mod 360 (mod 360 rawan terbalik
    #    saat sudutnya ada di sekitar 0/360 derajat).
    # =================================================================
    strike_180 = fault_strike % 180
    s1_180 = s1_trend % 180
    selisih = (s1_180 - strike_180) % 180
    sense = "right" if selisih <= 90 else "left"
    # Catatan kalibrasi: jika sense hasilnya TERBALIK secara konsisten
    # untuk seluruh data Anda (semua yang harusnya kanan keluar kiri,
    # dst), tinggal tukar baris di atas menjadi:
    # sense = "left" if selisih <= 90 else "right"
 
    # =================================================================
    # 2. REZIM TEGASAN (KOMPRESIONAL / EKSTENSIONAL) - kaidah Anderson
    #    dipakai KONSISTEN di semua band pitch (bukan cuma di 80-90).
    #    sigma1 curam -> ekstensional (normal); sigma3 curam -> kompresional
    # =================================================================
    regim_ekstensional = s1_plunge > s3_plunge
 
    # =================================================================
    # 3. EVALUASI MATRIKS 22 NAMA SESAR RICKARD (1971)
    # =================================================================
 
    # Pitch 0-10 (ujung horizontal - dominan strike-slip)
    if 0 <= pitch <= 10:
        if fault_dip >= 45:
            nama_sesar = "Right slip fault (No. 7)" if sense == "right" else "Left slip fault (No. 18)"
        else:
            nama_sesar = "Low-angle right slip fault" if sense == "right" else "Low-angle left slip fault"
 
    # Pitch 10-45
    elif 10 < pitch <= 45:
        if not regim_ekstensional:   # kompresional
            if fault_dip >= 45:
                nama_sesar = "Reverse right slip fault (No. 5)" if sense == "right" else "Reverse left slip fault (No. 22)"
            else:
                nama_sesar = "Thrust Right slip fault (No. 4)" if sense == "right" else "Thrust left slip fault (No. 19)"
        else:                        # ekstensional
            if fault_dip >= 45:
                nama_sesar = "Normal right slip fault (No. 11)" if sense == "right" else "Left Normal slip fault (No. 17)"
            else:
                nama_sesar = "Right lag slip fault (No. 9)" if sense == "right" else "Lag left slip fault (No. 14)"
 
    # Pitch 45-80
    elif 45 < pitch <= 80:
        if not regim_ekstensional:   # kompresional
            if fault_dip >= 45:
                nama_sesar = "Right reverse slip fault (No. 6)" if sense == "right" else "Left reverse slip fault (No. 21)"
            else:
                nama_sesar = "Right Thrust slip fault (No. 3)" if sense == "right" else "Left Thrust slip fault (No. 20)"
        else:                        # ekstensional
            if fault_dip >= 45:
                nama_sesar = "Right normal slip fault (No. 10)" if sense == "right" else "Normal left slip fault (No. 16)"
            else:
                nama_sesar = "Lag right slip fault (No. 8)" if sense == "right" else "Lag left slip fault (No. 15)"
 
    # Pitch 80-90 (ujung vertikal - dominan dip-slip)
    else:
        if regim_ekstensional:
            nama_sesar = "Normal slip fault (No. 13)" if fault_dip >= 45 else "Lag slip fault (No. 12)"
        else:
            nama_sesar = "Reverse slip fault (No. 2)" if fault_dip >= 45 else "Thrust slip fault (No. 1)"
 
    # =================================================================
    # 4. ARAH GAYA UTAMA (SIGMA 1) BERDASARKAN KUADRAN AZIMUTH
    # =================================================================
    s1_trend_360 = s1_trend % 360
    if (0 <= s1_trend_360 < 90) or (180 <= s1_trend_360 < 270):
        arah_gaya = "NE - SW"
    else:
        arah_gaya = "NW - SE"
 
    return nama_sesar, arah_gaya

# =====================================================================
# ANALISIS POPULASI & PEAK DENSITY
# =====================================================================

def find_dominant_plane(strike_arr, dip_arr, grid_step=4):
    """Mencari satu bidang dominan harga umum."""
    data_poles = np.array([strike_dip_to_pole(s, d) for s, d in zip(strike_arr, dip_arr)])
    best_s, best_d, best_density = 0, 0, -1
    for s in range(0, 360, grid_step):
        for d in range(0, 91, grid_step):
            p = strike_dip_to_pole(s, d)
            cos_ang = np.clip(data_poles @ p, -1, 1)
            density = np.sum(np.exp((cos_ang - 1) * 30))
            if density > best_density:
                best_s, best_d, best_density = s, d, density
    return best_s, best_d


def find_conjugate_shear_planes(strike_arr, dip_arr):
    """Memisahkan populasi data menjadi 2 kelompok menggunakan K-Means."""
    poles = np.array([strike_dip_to_pole(s, d) for s, d in zip(strike_arr, dip_arr)])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(poles)
    labels = kmeans.labels_
    
    idx1 = np.where(labels == 0)[0]
    idx2 = np.where(labels == 1)[0]
    
    s1, d1 = find_dominant_plane(strike_arr[idx1], dip_arr[idx1])
    s2, d2 = find_dominant_plane(strike_arr[idx2], dip_arr[idx2])
    
    return (s1, d1), (s2, d2), labels


# =====================================================================
# VISUALISASI STEREONET GASH & SHEAR CONJUGATE (TIDAK DIUBAH)
# =====================================================================

def plot_fracture_analysis(strike_arr, dip_arr, dom_strike, dom_dip, title):
    fig = plt.figure(figsize=(6, 6))
    if HAS_MPLSTEREONET:
        try:
            ax = fig.add_subplot(111, projection='stereonet')
            s_arr = np.array(strike_arr, dtype=float)
            d_arr = np.array(dip_arr, dtype=float)
            ax.pole(s_arr, d_arr, "o", color="#1f77b4", markersize=4, alpha=0.6, label="Pole Data")
            try:
                ax.density_contourf(s_arr, d_arr, measurement="poles", cmap="RdYlBu_r", alpha=0.5)
            except:
                pass
            ax.pole(dom_strike, dom_dip, "*", color="red", markersize=12, label="Pole Harga Umum")
            ax.plane(dom_strike, dom_dip, color="crimson", lw=2, label="Bidang Harga Umum")
            ax.pole(dom_strike, dom_dip, "o", mfc="none", mec="black", markersize=8, lw=1.5, label="Polar Titik Harga Umum")
            ax.grid(True)
            ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.35, 0.1))
            return fig
        except:
            pass
    ax = fig.add_subplot(111, projection='polar')
    return fig


def plot_conjugate_shear_stereonet(strike_arr, dip_arr, set1, set2, labels):
    fig = plt.figure(figsize=(6, 6))
    if HAS_MPLSTEREONET:
        try:
            ax = fig.add_subplot(111, projection='stereonet')
            s_arr, d_arr = np.array(strike_arr, dtype=float), np.array(dip_arr, dtype=float)
            ax.pole(s_arr, d_arr, "o", color="#1f77b4", markersize=3, alpha=0.6, label="Pole Data Mentah")
            try:
                ax.density_contourf(s_arr, d_arr, measurement="poles", cmap="RdYlBu_r", alpha=0.5)
            except:
                pass
            ax.pole(set1[0], set1[1], "*", color="red", markersize=10, label="Pole Harga Umum (Dua Set)")
            ax.pole(set2[0], set2[1], "*", color="red", markersize=10)
            ax.plane(set1[0], set1[1], color="crimson", lw=2, label="Bidang Harga Umum (Dua Set)")
            ax.plane(set2[0], set2[1], color="crimson", lw=2)
            ax.pole(set1[0], set1[1], "o", mfc="none", mec="black", markersize=8, lw=1.5, label="Polar Titik Harga Umum")
            ax.pole(set2[0], set2[1], "o", mfc="none", mec="black", markersize=8, lw=1.5)
            
            n1 = strike_dip_to_pole(set1[0], set1[1])
            n2 = strike_dip_to_pole(set2[0], set2[1])
            sig2_vec = np.cross(n1, n2)
            if np.linalg.norm(sig2_vec) > 1e-9:
                sig2_vec /= np.linalg.norm(sig2_vec)
                s2t, s2p = vector_to_trend_plunge(sig2_vec)
            ax.grid(True)
            ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.4, 0.1))
            return fig
        except:
            pass
    ax = fig.add_subplot(111, projection='polar')
    return fig


# =====================================================================
# VISUALISASI SUMBU TEGASAN TAB 4 (URUTAN PLUNGE, TREND YANG BENAR)
# =====================================================================

def plot_stress_stereonet(fault_strike, fault_dip, aux_strike, aux_dip, s1_tr, s1_pl, s2_tr, s2_pl, s3_tr, s3_pl, title):
    fig = plt.figure(figsize=(6, 6))
    if HAS_MPLSTEREONET:
        try:
            ax = fig.add_subplot(111, projection='stereonet')
            ax.plane(fault_strike, fault_dip, color="steelblue", lw=2, label="Bidang Sesar")
            ax.plane(aux_strike, aux_dip, color="seagreen", lw=2, ls="--", label="Bidang Bantu")
            
            ax.line(s1_pl, s1_tr, "rs", markersize=8, label=f"σ1 ({s1_tr:.0f}/{s1_pl:.0f})")
            ax.line(s2_pl, s2_tr, "o", color="orange", markersize=8, label=f"σ2 ({s2_tr:.0f}/{s2_pl:.0f})")
            ax.line(s3_pl, s3_tr, "^", color="purple", markersize=8, label=f"σ3 ({s3_tr:.0f}/{s3_pl:.0f})")
            ax.grid(True)
            ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.35, 1.1))
            return fig
        except:
            pass
    return fig


# =====================================================================
# REKONSTRUKSI STRUKTUR KINEMATIKA DAN PERHITUNGAN INTERSEKSI STRUKTUR
# =====================================================================

def calculate_kinematics_strictly_step_by_step(gash_dom, shear_set1, shear_set2, fault_strike, fault_dip):
    if not HAS_MPLSTEREONET:
        return None
    
    # 1. Sigma 2 = Perpotongan bidang Shear 1 dan Shear 2
    s2_pl, s2_tr = mplstereonet.plane_intersection(shear_set1[0], shear_set1[1], shear_set2[0], shear_set2[1])
    print(type(s2_tr))
    print(s2_tr)

    print(type(s2_pl))
    print(s2_pl)

    s2_tr = float(np.ravel(s2_tr)[0]) % 360
    s2_pl = float(np.ravel(s2_pl)[0])

    # 2. Bidang Bantu (Auxiliary Plane) adalah Polar dari Sigma 2
    # yang normalnya sama dengan arah sigma 2
    sigma2_vec = np.array([np.cos(np.radians(s2_pl)) * np.cos(np.radians(s2_tr)),np.cos(np.radians(s2_pl)) * np.sin(np.radians(s2_tr)),np.sin(np.radians(s2_pl))])
    aux_strike, aux_dip = normal_to_plane(sigma2_vec)

    # 3. Sigma 1 = Perpotongan Bidang Gash Fracture dan Bidang Bantu
    s1_pl, s1_tr = mplstereonet.plane_intersection(gash_dom[0], gash_dom[1], aux_strike, aux_dip)
    s1_tr = float(np.ravel(s1_tr)[0]) % 360
    s1_pl = float(np.ravel(s1_pl)[0])

    # 4. Sigma 3 = 90 derajat dari Sigma 1 sepanjang jalur lingkaran besar Bidang Bantu
    sigma1_vec = line_to_vector(s1_tr,s1_pl)
    sigma2_vec = line_to_vector(s2_tr,s2_pl)
    sigma3_vec = np.cross(sigma1_vec,sigma2_vec)
    sigma3_vec = sigma3_vec / np.linalg.norm(sigma3_vec)
    s3_tr, s3_pl = vector_to_trend_plunge(sigma3_vec)
    s3_tr = float(s3_tr) % 360
    s3_pl = float(s3_pl)

    # 5. Net Slip = Perpotongan antara Bidang Sesar Aktual dan Bidang Bantu
    ns_pl, ns_tr = mplstereonet.plane_intersection(fault_strike, fault_dip, aux_strike, aux_dip)
    ns_tr = float(np.ravel(ns_tr)[0]) % 360
    ns_pl = float(np.ravel(ns_pl)[0])

    # Jarak angular Pitch net slip pada permukaan bidang sesar menggunakan trigonometri vektor spasial murni
    strike_dir_r = np.radians(fault_strike)
    st_vec = np.array([np.cos(strike_dir_r), np.sin(strike_dir_r), 0.0])
    
    ns_pl_r = np.radians(ns_pl)
    ns_tr_r = np.radians(ns_tr)
    ns_vec = np.array([
        np.cos(ns_pl_r) * np.cos(ns_tr_r),
        np.cos(ns_pl_r) * np.sin(ns_tr_r),
        np.sin(ns_pl_r)
    ])
    
    raw_ang = angle_between_vectors(ns_vec, st_vec)
    pitch = float(raw_ang if raw_ang <= 90 else 180 - raw_ang)

    return {
        "fault_strike": fault_strike, "fault_dip": fault_dip,
        "aux_strike": aux_strike, "aux_dip": aux_dip,
        "sigma1": (s1_tr, s1_pl), "sigma2": (s2_tr, s2_pl), "sigma3": (s3_tr, s3_pl),
        "netslip_trend": ns_tr, "netslip_plunge": ns_pl, "pitch": pitch
    }


def plot_compilation_stereonet(gash_dom, shear_set1, shear_set2, kin_res):
    fig = plt.figure(figsize=(7, 7))
    if not HAS_MPLSTEREONET:
        ax = fig.add_subplot(111, projection='polar')
        return fig
        
    try:
        ax = fig.add_subplot(111, projection='stereonet')
        
        # Menggambar Lingkaran Besar Bidang Struktur Utama
        ax.plane(kin_res["fault_strike"], kin_res["fault_dip"], color="black", lw=2.5, label="Bidang Sesar")
        ax.plane(kin_res["aux_strike"], kin_res["aux_dip"], color="dimgray", lw=1.2, ls=":", label="Bidang Bantu")
        
        if gash_dom:
            ax.plane(gash_dom[0], gash_dom[1], color="blue", lw=1.2, ls="--", label="Bidang Gash")
        if shear_set1:
            ax.plane(shear_set1[0], shear_set1[1], color="red", lw=1.2, label="Bidang Shear 1")
        if shear_set2:
            ax.plane(shear_set2[0], shear_set2[1], color="green", lw=1.2, label="Bidang Shear 2")
            
        # Plotting Titik Tunggal Murni (Plunge, Trend)
        s1_trend, s1_plunge = kin_res["sigma1"]
        s2_trend, s2_plunge = kin_res["sigma2"]
        s3_trend, s3_plunge = kin_res["sigma3"]
        ns_trend, ns_plunge = kin_res["netslip_trend"], kin_res["netslip_plunge"]
        
        ax.line(s1_plunge, s1_trend, "s", color="red", markersize=9, label=f"σ1 ({s1_trend:.0f}/{s1_plunge:.0f})")
        ax.line(s2_plunge, s2_trend, "o", color="orange", markersize=9, label=f"σ2 ({s2_trend:.0f}/{s2_plunge:.0f})")
        ax.line(s3_plunge, s3_trend, "^", color="purple", markersize=9, label=f"σ3 ({s3_trend:.0f}/{s3_plunge:.0f})")
        ax.line(ns_plunge, ns_trend, "X", color="cyan", markersize=10, mec="black", label=f"Net Slip ({ns_trend:.0f}/{ns_plunge:.0f})")
        
        ax.grid(True, color="gainsboro", linestyle="-", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.35, 0.0))
        ax.set_title("Kompilasi Data Stereonet Akhir Gabungan", fontsize=11, fontweight='bold', pad=15)
        return fig
    except Exception as e:
        st.error(f"Gagal memplot kompilasi akhir: {str(e)}")
        return fig


# =====================================================================
# MAIN APPLICATION INTERFACE
# =====================================================================

def main():
    with st.sidebar:
        st.header("⚙️ Panel Kontrol Input")
        uploaded_file = st.file_uploader("Upload File Data Lapangan", type=["csv", "txt"])

    st.title("FAULTGRAM")
    st.subheader("Analisis Sesar Berdasarkan Klasifikasi Rickard 1971")
    st.markdown("---")

    if uploaded_file is None:
        st.info("👈 Silakan upload file CSV/TXT data lapangan pada sidebar.")
        return

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        st.error(f"Gagal membaca file: {str(e)}")
        return

    with st.expander("🔧 Pemetaan Kolom Data", expanded=True):
        cols = ["(Tidak ada)"] + list(df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            gash_strike_col = st.selectbox("Gash Fracture - Strike", cols, key="gs")
            gash_dip_col = st.selectbox("Gash Fracture - Dip", cols, key="gd")
        with c2:
            shear_strike_col = st.selectbox("Shear Fracture - Strike", cols, key="ss")
            shear_dip_col = st.selectbox("Shear Fracture - Dip", cols, key="sd")
        with c3:
            fault_strike_col = st.selectbox("Bidang Sesar - Strike", cols, key="fls")
            fault_dip_col = st.selectbox("Bidang Sesar - Dip", cols, key="fld")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Viewer", "🔹 Gash Fracture", "🔸 Shear Fracture (Conjugate)", "🧭 Sumbu Tegasan & Kinematika", "🏷️ Hasil Analisis Kompilasi Akhir"
    ])

    with tab1:
        st.dataframe(df, use_container_width=True)

    gash_dom = None
    with tab2:
        st.subheader("🔹 Analisis Gash Fracture")
        if gash_strike_col != "(Tidak ada)" and gash_dip_col != "(Tidak ada)":
            sub = df[[gash_strike_col, gash_dip_col]].apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = ["strike", "dip"]
            if len(sub) > 0:
                dom_s, dom_d = find_dominant_plane(sub["strike"], sub["dip"])
                gash_dom = (dom_s, dom_d)
                st.metric("Gash Strike/Dip Harga Umum", f"{dom_s:.0f}°/{dom_d:.0f}°")
                fig = plot_fracture_analysis(sub["strike"], sub["dip"], dom_s, dom_d, "Gash Plane Analysis")
                if fig is not None:
                    st.pyplot(fig)

    shear_set1, shear_set2 = None, None
    with tab3:
        st.subheader("🔸 Analisis Shear Fracture Konjugasi (Poin 2 - Dua Set Harga Umum)")
        if shear_strike_col != "(Tidak ada)" and shear_dip_col != "(Tidak ada)":
            sub = df[[shear_strike_col, shear_dip_col]].apply(pd.to_numeric, errors="coerce").dropna()
            sub.columns = ["strike", "dip"]
            if len(sub) > 0:
                set1, set2, labels = find_conjugate_shear_planes(sub["strike"].values, sub["dip"].values)
                shear_set1, shear_set2 = set1, set2
                
                n1 = strike_dip_to_pole(set1[0], set1[1])
                n2 = strike_dip_to_pole(set2[0], set2[1])
                sudut_konj = angle_between_vectors(n1, n2)
                if sudut_konj > 90: sudut_konj = 180 - sudut_konj

                cc1, cc2, cc3 = st.columns(3)
                with cc1: st.metric("Strike/Dip Harga Umum Set 1", f"{set1[0]:.0f}° / {set1[1]:.0f}°")
                with cc2: st.metric("Strike/Dip Harga Umum Set 2", f"{set2[0]:.0f}° / {set2[1]:.0f}°")
                with cc3: st.metric("Sudut Konjugasi", f"{sudut_konj:.0f}°")

                fig_conj = plot_conjugate_shear_stereonet(sub["strike"], sub["dip"], set1, set2, labels)
                if fig_conj is not None:
                    st.pyplot(fig_conj)

    kinematics_result = None
    with tab4:
        st.subheader("🧭 Sumbu Tegasan & Kinematika")
        if gash_dom and shear_set1 and shear_set2 and fault_strike_col != "(Tidak ada)" and fault_dip_col != "(Tidak ada)":
            f_sub = df[[fault_strike_col, fault_dip_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(f_sub) > 0:
                f_strike = float(f_sub.iloc[0].iloc[0])
                f_dip = float(f_sub.iloc[0].iloc[1])
                
                kinematics_result = calculate_kinematics_strictly_step_by_step(gash_dom, shear_set1, shear_set2, f_strike, f_dip)
                if kinematics_result is not None:
                    fig = plot_stress_stereonet(f_strike, f_dip, kinematics_result["aux_strike"], kinematics_result["aux_dip"],
                                                kinematics_result["sigma1"][0], kinematics_result["sigma1"][1],
                                                kinematics_result["sigma2"][0], kinematics_result["sigma2"][1],
                                                kinematics_result["sigma3"][0], kinematics_result["sigma3"][1],
                                                "Stress Tensor Configuration")
                    if fig is not None:
                        st.pyplot(fig)

    with tab5:
        st.subheader("🏷️ Hasil Final Analisis Kompilasi Gabungan")
        if kinematics_result is not None:
            # Pengepakan seluruh parameter kinematika asli Anda
            fault_strike = kinematics_result["fault_strike"]
            f_dip = kinematics_result["fault_dip"]
            pitch = kinematics_result["pitch"]
            s1_tr, s1_pl = kinematics_result["sigma1"]
            s2_tr, s2_pl = kinematics_result["sigma2"]
            s3_tr, s3_pl = kinematics_result["sigma3"]
            
            # Pemanggilan fungsi klasifikasi Rickard (1971) yang sudah diperbaiki orientasi pitch-nya
            nama_rickard, arah_gaya_utama = klasifikasi_sesar_rickard_1971(
                fault_strike, f_dip, pitch, s1_tr, s1_pl, s3_pl, s2_pl
            )
            arah_gaya_str = hitung_arah_mata_angin(s1_tr)
            
            # Menampilkan Header Hasil Akhir Presisi
            st.success(f"### 📑 Klasifikasi Sesar Rickard (1971): **{nama_rickard}**")
            
            # Tampilan Kolom Metric Parameter Utama
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Arah Gaya Utama (σ1)", arah_gaya_utama)
            with c2:
                st.metric("Kemiringan Sesar (Dip)", f"{f_dip:.1f}°")
            with c3:
                st.metric("Nilai Pitch Slickenside", f"{pitch:.1f}°")
                
            st.markdown("---")
            
            # Tampilkan Nilai Posisi Sumbu Tegasan Spasial
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Kedudukan Sesar Aktual:** N {fault_strike:.0f}° E / {f_dip:.0f}°")
                st.warning(f"**Kedudukan Sumbu Sesar σ1:** {s1_tr:.1f}° / {s1_pl:.1f}° ({arah_gaya_str})")
            with col2:
                st.warning(f"**Kedudukan Sumbu Sesar σ2 (Intersection):** {s2_tr:.1f}° / {s2_pl:.1f}°")
                st.warning(f"**Kedudukan Sumbu Sesar σ3:** {s3_tr:.1f}° / {s3_pl:.1f}°")
                
            st.markdown("---")
            
            # Menampilkan Plotingan Stereonet Akhir Gabungan (Aman tanpa perubahan)
            fig_compile = plot_compilation_stereonet(gash_dom, shear_set1, shear_set2, kinematics_result)
            if fig_compile is not None:
                st.pyplot(fig_compile)
        else:
            st.info("Lengkapi analisis pada Tab 'Sumbu Tegasan & Kinematika' terlebih dahulu.")

if __name__ == "__main__":
    main()