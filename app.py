import streamlit as st
import requests
import pandas as pd
import datetime
import hashlib
import psycopg2
from psycopg2 import IntegrityError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. KONFIGURASI KURS, SESI & KONEKSI POSTGRESQL
# ==========================================
KURS_RUPIAH = 18000

# Inisialisasi Session State untuk Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Fungsi Enkripsi Password (SHA-256)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Fungsi Bantuan untuk Membuat Koneksi ke PostgreSQL
def get_db_connection():
    try:
        # Pastikan Anda telah mengatur DB_URL di .streamlit/secrets.toml
        return psycopg2.connect(st.secrets["DB_URL"])
    except Exception as e:
        st.error(f"⚠️ Koneksi ke database gagal: {e}")
        return None

# Fungsi inisialisasi tabel Database (Users & User Wishlist)
def init_db():
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as c:
            # Tabel Pengguna
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR(255) PRIMARY KEY,
                    password TEXT NOT NULL
                );
            ''')
            # Tabel Wishlist berbasis Username
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_wishlist (
                    username VARCHAR(255),
                    game_id VARCHAR(100),
                    judul TEXT,
                    gambar TEXT,
                    PRIMARY KEY (username, game_id),
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                );
            ''')
            conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"⚠️ Gagal membuat tabel di PostgreSQL: {e}")
    finally:
        conn.close()

# Jalankan inisialisasi tabel saat aplikasi dimulai
init_db()

# ==========================================
# 1. FUNGSI CRUD DATABASE & OTENTIKASI
# ==========================================
def daftar_user(username, password):
    conn = get_db_connection()
    if conn is None:
        return False, "⚠️ Gagal terhubung ke database server."
    try:
        with conn.cursor() as c:
            c.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hash_password(password)))
            conn.commit()
        return True, "✅ Registrasi berhasil! Silakan login pada tab Login."
    except IntegrityError:
        conn.rollback()
        return False, "⚠️ Username sudah digunakan. Silakan pilih username lain."
    except Exception as e:
        conn.rollback()
        return False, f"⚠️ Terjadi kesalahan sistem: {e}"
    finally:
        conn.close()

def cek_login(username, password):
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        with conn.cursor() as c:
            c.execute("SELECT username FROM users WHERE username=%s AND password=%s", (username, hash_password(password)))
            user = c.fetchone()
        return user is not None
    except Exception:
        return False
    finally:
        conn.close()

def tambah_ke_wishlist(username, game_id, judul, gambar):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO user_wishlist (username, game_id, judul, gambar) VALUES (%s, %s, %s, %s)", 
                (username, str(game_id), judul, gambar)
            )
            conn.commit()
        st.toast(f"✅ '{judul}' berhasil disimpan ke Wishlist kamu!", icon="💾")
    except IntegrityError:
        conn.rollback()
        st.toast(f"⚠️ '{judul}' sudah ada di dalam Wishlist kamu!", icon="ℹ️")
    except Exception as e:
        conn.rollback()
        st.error(f"Gagal menyimpan ke wishlist: {e}")
    finally:
        conn.close()

def ambil_wishlist(username):
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        with conn.cursor() as c:
            c.execute("SELECT game_id, judul, gambar FROM user_wishlist WHERE username=%s", (username,))
            rows = c.fetchall()
            if not rows:
                return pd.DataFrame()
            colnames = [desc[0] for desc in c.description]
            return pd.DataFrame(rows, columns=colnames)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def hapus_dari_wishlist(username, game_id):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM user_wishlist WHERE username=%s AND game_id=%s", (username, str(game_id)))
            conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Gagal menghapus data: {e}")
    finally:
        conn.close()

# ==========================================
# 2. PENGATURAN HALAMAN WEB & TEMA
# ==========================================
st.set_page_config(page_title="Pencari Diskon Game", page_icon="🎮", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #121212; }
    h1, h2 { color: #00ffcc; font-family: 'Segoe UI', sans-serif; }
    h3, h4, h5 { color: #ffffff; }
    .stMetric { background-color: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 4px solid #00ffcc; margin-bottom: 10px; }
    .rec-box { background-color: #1a1a2e; padding: 12px; border-radius: 8px; border: 1px solid #00ffcc; margin-bottom: 10px; min-height: 420px; }
    hr { border-color: #262730; }
    </style>
""", unsafe_allow_html=True)

STORE_MAPPING = {
    "1": "Steam", "2": "GamersGate", "3": "GreenManGaming", "7": "GOG", 
    "11": "Humble Store", "25": "Epic Games", "30": "IndieGala"
}

# ==========================================
# 3. FUNGSI AMBIL DATA DARI SERVER (API)
# ==========================================
# Tambahkan User-Agent agar tidak diblokir oleh Cloudflare/CheapShark
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

@st.cache_data(ttl=300)  # Menurunkan TTL cache jadi 5 menit agar cepat update
def fetch_deals():
    url = "https://www.cheapshark.com/api/1.0/deals"
    params = {"sortBy": "Savings", "onSale": 1, "pageSize": 150}
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"⚠️ API CheapShark merespons dengan status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"⚠️ Kesalahan koneksi ke CheapShark: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_game_prices(game_id):
    url = f"https://www.cheapshark.com/api/1.0/games?id={game_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}

with st.spinner("Sedang menyinkronkan data diskon terbaru..."):
    raw_data = fetch_deals()

df = pd.DataFrame()
if raw_data:
    processed_data = []
    waktu_sekarang = datetime.datetime.now()
    
    for item in raw_data: 
        nama_toko = STORE_MAPPING.get(str(item['storeID']), f"Toko Lain ({item['storeID']})")
        link_toko = f"https://www.cheapshark.com/redirect?dealID={item['dealID']}"
        
        timestamp_diskon = int(item.get('lastChange', 0))
        waktu_mulai_diskon = datetime.datetime.fromtimestamp(timestamp_diskon)
        selisih_waktu = waktu_sekarang - waktu_mulai_diskon
        total_jam_aktif = selisih_waktu.total_seconds() / 3600
        
        if total_jam_aktif < 24:
            status_urgensi = "🔥 Baru (< 24 Jam)"
        elif total_jam_aktif < 72:
            status_urgensi = "⚡ Stabil (1-3 Hari)"
        else:
            status_urgensi = "⚠️ Lama (> 3 Hari)"
            
        harga_usd = float(item['salePrice'])
        
        processed_data.append({
            "Game_ID": item['gameID'],
            "Gambar": item['thumb'], 
            "Judul Game": item['title'],
            "Toko": nama_toko,
            "Harga Diskon ($)": harga_usd,
            "Estimasi Rupiah (Rp)": harga_usd * KURS_RUPIAH,
            "Harga Normal": float(item['normalPrice']),
            "Harga_Normal_Format": float(item['normalPrice']),
            "Hemat": f"{float(item['savings']):.0f}%",
            "Mulai Diskon": waktu_mulai_diskon.strftime("%d %b %Y, %H:%M"),
            "Status Diskon": status_urgensi,
            "Link Beli": link_toko,
            "Metacritic": int(item.get('metacriticScore', 0)),
            "Skor Kelayakan (0-10)": float(item.get('dealRating', 0)),
            "Fitur_Konten": f"{item['title']} {nama_toko}"
        })
    df = pd.DataFrame(processed_data)
    
# ==========================================
# 4. TOMBOL MENU NAVIGASI (SIDEBAR)
# ==========================================
st.sidebar.title("🎮 Menu Aplikasi")

# Tampilan Profil Pengguna di Sidebar
if st.session_state['logged_in']:
    st.sidebar.success(f"👤 Login sebagai: **{st.session_state['username']}**")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.toast("Anda telah berhasil logout.", icon="ℹ️")
        st.rerun()
else:
    st.sidebar.info("🔒 Anda belum login. Masuk untuk menggunakan fitur Wishlist.")

st.sidebar.markdown("---")

menu_pilihan = st.sidebar.radio(
    "PILIH HALAMAN:",
    [
        "🎯 Semua Diskon Utama",
        "💖 Wishlist Saya",
        "📊 Bandingkan Harga Toko",
        "🎁 Pojok Pemburu Diskon",
        "🤖 Cari Game Serupa",
        "💰 Racik Paket Budget",
        "👥 Patungan Mabar"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data terintegrasi real-time & Database cloud PostgreSQL.")

# ==========================================
# 5. LOGIKA PERPINDAHAN HALAMAN
# ==========================================
if df.empty:
    st.error("⚠️ Gagal memuat data dari API CheapShark. Periksa koneksi internet atau coba kembali beberapa saat lagi.")
else:
    # ------------------------------------------
    # MENU 1: SEMUA DISKON UTAMA
    # ------------------------------------------
    if menu_pilihan == "🎯 Semua Diskon Utama":
        st.title("🎯 Daftar Diskon Utama & Filter Kualitas")
        
        with st.expander("🛠️ Panel Filter Canggih", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                search_query = st.text_input("🔍 Cari judul game:", "")
                selected_store = st.selectbox("🏬 Filter Toko:", ["Semua Toko"] + list(STORE_MAPPING.values()))
            with col2:
                min_metacritic = st.slider("⭐ Minimal Skor Metacritic:", 0, 100, 0, 5)
                min_deal_rating = st.slider("📈 Minimal Skor Kelayakan:", 0.0, 10.0, 0.0, 0.5)
            
        df_filtered = df.copy()
        if search_query:
            df_filtered = df_filtered[df_filtered["Judul Game"].str.contains(search_query, case=False)]
        if selected_store != "Semua Toko":
            df_filtered = df_filtered[df_filtered["Toko"] == selected_store]
            
        df_filtered = df_filtered[
            (df_filtered["Metacritic"] >= min_metacritic) & 
            (df_filtered["Skor Kelayakan (0-10)"] >= min_deal_rating)
        ]
        
        st.markdown("---")
        
        if df_filtered.empty:
            st.info("Tidak ada game yang cocok dengan kriteria filter saat ini.")
        else:
            total_data = len(df_filtered)
            batas_min = 1 if total_data > 0 else 0
            limit_tampil = st.number_input("Tampilkan jumlah game:", min_value=batas_min, max_value=total_data, value=min(20, total_data), step=1)
            df_display = df_filtered.head(limit_tampil)
            
            hdr_cols = st.columns([1, 3.5, 1.5, 2, 1.5, 1, 1])
            with hdr_cols[0]: st.markdown("**Tampilan**")
            with hdr_cols[1]: st.markdown("**Judul Game**")
            with hdr_cols[2]: st.markdown("**Toko**")
            with hdr_cols[3]: st.markdown("**Harga (Rupiah)**")
            with hdr_cols[4]: st.markdown("**Kualitas**")
            with hdr_cols[5]: st.markdown("**Beli**")
            with hdr_cols[6]: st.markdown("**Simpan**")
            st.markdown("<hr style='margin: 0 0 10px 0; border: 1px solid #00ffcc;'>", unsafe_allow_html=True)
            
            for i, (_, row) in enumerate(df_display.iterrows()):
                row_cols = st.columns([1, 3.5, 1.5, 2, 1.5, 1, 1])
                
                with row_cols[0]:
                    st.image(row["Gambar"], use_container_width=True)
                    
                with row_cols[1]:
                    st.markdown(f"**{row['Judul Game']}**")
                    st.caption(f"{row['Status Diskon']}")
                    
                with row_cols[2]:
                    st.write(row["Toko"])
                    
                with row_cols[3]:
                    if row["Harga Diskon ($)"] == 0:
                        st.markdown("<span style='color:#ff4b4b; font-weight:bold;'>GRATIS! 🎁</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"~~Rp {int(row['Harga Normal'] * KURS_RUPIAH):,}~~")
                        st.markdown(f"**Rp {int(row['Estimasi Rupiah (Rp)']):,}** <span style='color:#00ffcc; font-size:12px;'>({row['Hemat']})</span>", unsafe_allow_html=True)
                        
                with row_cols[4]:
                    st.markdown(f"⭐ {row['Metacritic']} <br> 📈 {row['Skor Kelayakan (0-10)']:.1f}", unsafe_allow_html=True)
                    
                with row_cols[5]:
                    st.link_button("🛒", row["Link Beli"], use_container_width=True, help="Buka halaman promo toko resmi")
                    
                with row_cols[6]:
                    unik_key = f"btn_wish_{row['Game_ID']}_{row['Toko']}_{i}"
                    if st.button("➕", key=unik_key, use_container_width=True, help="Masukkan game ini ke Wishlist"):
                        if st.session_state['logged_in']:
                            tambah_ke_wishlist(st.session_state['username'], row["Game_ID"], row["Judul Game"], row["Gambar"])
                        else:
                            st.toast("⚠️ Silakan login di menu '💖 Wishlist Saya' terlebih dahulu untuk menyimpan game!", icon="🔒")
                        
                st.markdown("<hr style='margin: 8px 0; border: 0.5px solid #262730;'>", unsafe_allow_html=True)

    # ------------------------------------------
    # MENU 2: WISHLIST
    # ------------------------------------------
    elif menu_pilihan == "💖 Wishlist Saya":
        st.title("💖 Wishlist Game Incaran")
        
        if not st.session_state['logged_in']:
            st.info("🔒 **Halaman Khusus Member.** Silakan login atau buat akun baru terlebih dahulu untuk mengakses fitur Wishlist.")
            
            tab_login, tab_daftar = st.tabs(["🔑 Login Akun", "📝 Daftar Akun Baru"])
            
            with tab_login:
                st.write("### Masuk ke Akun Kamu")
                login_user = st.text_input("Username:", key="login_user")
                login_pass = st.text_input("Password:", type="password", key="login_pass")
                
                if st.button("Masuk 🚀", use_container_width=True):
                    if login_user and login_pass:
                        if cek_login(login_user, login_pass):
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = login_user
                            st.success(f"Selamat datang kembali, {login_user}!")
                            st.rerun()
                        else:
                            st.error("❌ Username atau Password salah.")
                    else:
                        st.warning("⚠️ Harap isi username dan password.")
                        
            with tab_daftar:
                st.write("### Buat Akun Baru")
                reg_user = st.text_input("Pilih Username Baru:", key="reg_user")
                reg_pass = st.text_input("Pilih Password:", type="password", key="reg_pass")
                reg_pass_conf = st.text_input("Ulangi Password:", type="password", key="reg_pass_conf")
                
                if st.button("Daftar Sekarang 📋", use_container_width=True):
                    if reg_user and reg_pass:
                        if reg_pass == reg_pass_conf:
                            sukses, pesan = daftar_user(reg_user, reg_pass)
                            if sukses:
                                st.success(pesan)
                            else:
                                st.error(pesan)
                        else:
                            st.warning("⚠️ Konfirmasi password tidak cocok.")
                    else:
                        st.warning("⚠️ Semua kolom wajib diisi.")

        else:
            st.markdown(f"### *Selamat datang di koleksi game impianmu, **{st.session_state['username']}**!*")
            
            db_wishlist = ambil_wishlist(st.session_state['username'])
            
            if db_wishlist.empty:
                st.info("Wishlist kamu masih kosong. Silakan tambahkan game incaran dari menu '🎯 Semua Diskon Utama'.")
            else:
                st.success(f"Terdapat **{len(db_wishlist)} game** di dalam wishlist pribadi kamu.")
                
                kolom_per_baris = 3
                for i in range(0, len(db_wishlist), kolom_per_baris):
                    cols = st.columns(kolom_per_baris)
                    for j, col in enumerate(cols):
                        if i + j < len(db_wishlist):
                            row = db_wishlist.iloc[i + j]
                            game_id_wishlist = row['game_id']
                            
                            with col:
                                st.markdown(f"""
                                <div style="background-color:#1a1a2e; padding:15px; border-radius:10px; border:1px solid #00ffcc; margin-bottom:15px; text-align:center;">
                                    <img src="{row['gambar']}" style="width:100%; height:120px; object-fit:cover; border-radius:8px; margin-bottom:10px;">
                                    <h4 style="color:#ffffff; font-size:16px; margin:0 0 10px 0;">{row['judul']}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.spinner("Mengecek harga live..."):
                                    live_data = fetch_game_prices(game_id_wishlist)
                                
                                if live_data and 'deals' in live_data and len(live_data['deals']) > 0:
                                    termurah = live_data['deals'][0] 
                                    toko_termurah = STORE_MAPPING.get(str(termurah['storeID']), "Toko Lain")
                                    harga_sekarang_rp = float(termurah['price']) * KURS_RUPIAH
                                    link_redirect = f"https://www.cheapshark.com/redirect?dealID={termurah['dealID']}"
                                    
                                    st.markdown(f"🏆 Termurah di **{toko_termurah}**")
                                    st.markdown(f"💰 **Rp {int(harga_sekarang_rp):,}**")
                                    st.link_button("Beli Sekarang 🛒", link_redirect, use_container_width=True)
                                else:
                                    st.warning("Sedang tidak ada promo untuk game ini.")
                                    
                                if st.button("🗑️ Hapus", key=f"del_{game_id_wishlist}_{i+j}", use_container_width=True):
                                    hapus_dari_wishlist(st.session_state['username'], game_id_wishlist)
                                    st.rerun()

    # ------------------------------------------
    # MENU 3: BANDINGKAN HARGA TOKO
    # ------------------------------------------
    elif menu_pilihan == "📊 Bandingkan Harga Toko":
        st.title("📊 Matriks Perbandingan Harga Antar Retail")
        
        pilihan_game_banding = st.selectbox("🎯 Pilih Game yang Ingin Dikomparasi:", sorted(df["Judul Game"].unique()))
        
        if pilihan_game_banding:
            game_id_target = df[df["Judul Game"] == pilihan_game_banding]["Game_ID"].values[0]
            with st.spinner("Membandingkan harga di berbagai toko..."):
                detail_game = fetch_game_prices(game_id_target)
                
            if detail_game and 'deals' in detail_game:
                data_komparasi = []
                for deal in detail_game['deals']:
                    data_komparasi.append({
                        "Nama Toko": STORE_MAPPING.get(str(deal['storeID']), f"Toko ID {deal['storeID']}"),
                        "Harga Diskon (Rp)": float(deal['price']) * KURS_RUPIAH,
                        "Harga Normal (Rp)": float(deal['retailPrice']) * KURS_RUPIAH,
                        "Link": f"https://www.cheapshark.com/redirect?dealID={deal['dealID']}"
                    })
                df_komparasi = pd.DataFrame(data_komparasi)
                
                g_col1, g_col2 = st.columns([3, 2])
                with g_col1:
                    st.markdown("#### 📈 Grafik Perbandingan Harga Diskon (Rp):")
                    st.bar_chart(df_komparasi, x="Nama Toko", y="Harga Diskon (Rp)", color="#00ffcc")
                with g_col2:
                    st.markdown("#### 📋 Rincian Transparansi Harga:")
                    st.dataframe(
                        df_komparasi, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Harga Diskon (Rp)": st.column_config.NumberColumn(format="Rp %,.0f"),
                            "Harga Normal (Rp)": st.column_config.NumberColumn(format="Rp %,.0f"),
                            "Link": st.column_config.LinkColumn("Beli", display_text="Ke Toko 🛒")
                        }
                    )
            else:
                st.warning("Data komparasi toko alternatif tidak ditemukan.")

    # ------------------------------------------
    # MENU 4: POJOK PEMBURU DISKON
    # ------------------------------------------
    elif menu_pilihan == "🎁 Pojok Pemburu Diskon":
        st.title("🎁 Pojok Pemburu Diskon")
        
        df_gratis = df[df["Harga Diskon ($)"] == 0]
        df_murah = df[(df["Estimasi Rupiah (Rp)"] > 0) & (df["Estimasi Rupiah (Rp)"] <= 50000)].sort_values("Estimasi Rupiah (Rp)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🆓 Game Gratis Saat Ini (Rp 0)")
            if not df_gratis.empty:
                st.dataframe(
                    df_gratis[["Judul Game", "Toko", "Link Beli"]], 
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Link Beli": st.column_config.LinkColumn("Aksi", display_text="Klaim 🎁")
                    }
                )
            else:
                st.info("Saat ini belum ada game gratis yang tersedia.")
                
        with c2:
            st.subheader("🪙 Game di Bawah Rp 50.000")
            if not df_murah.empty:
                st.dataframe(
                    df_murah[["Judul Game", "Toko", "Estimasi Rupiah (Rp)", "Link Beli"]], 
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Estimasi Rupiah (Rp)": st.column_config.NumberColumn(format="Rp %,.0f"),
                        "Link Beli": st.column_config.LinkColumn("Beli", display_text="🛒")
                    }
                )
            else:
                st.info("Tidak ditemukan game di bawah Rp 50.000.")

    # ------------------------------------------
    # MENU 5: CARI GAME SERUPA (AI)
    # ------------------------------------------
    elif menu_pilihan == "🤖 Cari Game Serupa":
        st.title("🤖 Cari Alternatif Game Serupa")
        
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            game_target = st.selectbox("Pilih Game Target:", sorted(df["Judul Game"].unique()))
        with ai_c2:
            ambang = st.slider("Batas Kemiripan (%):", 1, 100, 15)
        
        if game_target:
            try:
                idx = df[df["Judul Game"] == game_target].index[0]
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(df["Fitur_Konten"])
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                skor = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
                rekomendasi_idx = [i[0] for i in skor if i[0] != idx and i[1] >= ambang/100.0]
                
                if rekomendasi_idx:
                    df_rek = df.iloc[rekomendasi_idx]
                    st.dataframe(
                        df_rek[["Gambar", "Judul Game", "Toko", "Estimasi Rupiah (Rp)"]], 
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Gambar": st.column_config.ImageColumn("Preview"),
                            "Estimasi Rupiah (Rp)": st.column_config.NumberColumn(format="Rp %,.0f")
                        }
                    )
                else:
                    st.info("Tidak ada game yang cukup mirip dengan ambang batas tersebut. Coba turunkan persentase kemiripan.")
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam memproses rekomendasi AI: {e}")

    # ------------------------------------------
    # MENU 6: RACIK PAKET BUDGET
    # ------------------------------------------
    elif menu_pilihan == "💰 Racik Paket Budget":
        st.title("💰 Racik Paket Game Otomatis")
        budget = st.number_input("Masukkan Budget (Rp):", min_value=10000, max_value=5000000, value=150000, step=10000)
        budget_usd = budget / KURS_RUPIAH
        
        if st.button("Racik Sekarang 🚀", use_container_width=True):
            df_terjangkau = df[df["Harga Diskon ($)"] <= budget_usd].sort_values(by='Skor Kelayakan (0-10)', ascending=False)
            keranjang = []
            total = 0.0
            
            for _, row in df_terjangkau.iterrows():
                if total + row['Harga Diskon ($)'] <= budget_usd:
                    keranjang.append(row)
                    total += row['Harga Diskon ($)']
                    
            if keranjang:
                st.success(f"🎉 Berhasil mendapatkan **{len(keranjang)} Game**! Total: **Rp {int(total * KURS_RUPIAH):,}** (Sisa budget: Rp {int((budget_usd - total) * KURS_RUPIAH):,})")
                st.dataframe(
                    pd.DataFrame(keranjang)[["Judul Game", "Toko", "Estimasi Rupiah (Rp)", "Link Beli"]], 
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Estimasi Rupiah (Rp)": st.column_config.NumberColumn(format="Rp %,.0f"),
                        "Link Beli": st.column_config.LinkColumn("Beli", display_text="🛒")
                    }
                )
            else:
                st.error("Budget tidak cukup untuk membeli game apa pun pada diskon saat ini. Coba naikkan budget Anda!")

    # ------------------------------------------
    # MENU 7: PATUNGAN MABAR
    # ------------------------------------------
    elif menu_pilihan == "👥 Patungan Mabar":
        st.title("👥 Kalkulator Patungan Mabar")
        st.write("Hitung biaya pembagian jika kamu ingin membeli paket game *multiplayer* atau *family sharing* bersama teman-temanmu!")
        
        c_patungan1, c_patungan2 = st.columns(2)
        with c_patungan1:
            harga = st.number_input("Total Harga Game/Paket (Rp):", min_value=0, max_value=10000000, value=150000, step=5000)
        with c_patungan2:
            orang = st.slider("Jumlah Orang:", min_value=2, max_value=10, value=3)
            
        if harga > 0:
            biaya_per_orang = int(harga / orang)
            st.metric("Biaya yang Harus Dibayar Per Orang:", f"Rp {biaya_per_orang:,}")
            st.info(f"💡 Tip: Dengan membagi kepada {orang} orang, kalian masing-masing menghemat **Rp {int(harga - biaya_per_orang):,}** dibandingkan beli sendiri!")
