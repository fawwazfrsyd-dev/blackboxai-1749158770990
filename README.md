# XAUUSD Trading Bot README

Selamat datang di **XAUUSD Trading Bot**, sebuah bot Telegram berbasis AI yang dirancang untuk memberikan sinyal trading XAUUSD menggunakan analisis teknikal, fundamental, dan machine learning. Bot ini dikembangkan oleh xAI dan dioptimalkan untuk berjalan pada server CPU-only, khususnya di VPS.

## Persyaratan
- **Python**: Versi 3.8 atau lebih baru.
- **Sistem Operasi**: Ubuntu, Debian, CentOS, atau distribusi Linux lain yang umum digunakan di VPS.
- **Dependensi**: Lihat `requirements.txt` untuk daftar lengkap.
- **Koneksi Internet**: Stabil, untuk streaming data real-time dan API.
- **VPS**: Minimal 1 vCPU, 8GB RAM, dan 20GB penyimpanan (direkomendasikan).

## Fitur Utama
- Analisis teknikal dengan lebih dari 40 indikator (RSI, MACD, Fibonacci, dll.).
- Analisis fundamental (berita, sentimen X, kalender ekonomi, dll.).
- Prediksi harga menggunakan GNN dan Transformer.
- Manajemen risiko dengan Monte Carlo simulation.
- Visualisasi sinyal dan dashboard trading.
- Simulasi trade dan backtesting.
- Notifikasi otomatis untuk event ekonomi dan volatilitas tinggi.

## Instalasi di VPS

### 1. Persiapan Lingkungan
1. **Koneksi ke VPS**:
   - Gunakan SSH untuk masuk ke VPS:
     ```
     ssh user@your_vps_ip
     ```
   - Ganti `user` dengan nama pengguna VPS Anda dan `your_vps_ip` dengan alamat IP VPS.

2. **Perbarui Sistem**:
   - Pastikan sistem VPS diperbarui:
     ```
     sudo apt update && sudo apt upgrade -y
     ```

3. **Instal Python dan Dependensi Dasar**:
   - Instal Python dan pip:
     ```
     sudo apt install python3 python3-pip -y
     ```
   - Verifikasi instalasi:
     ```
     python3 --version
     pip3 --version
     ```

4. **Buat Direktori Proyek**:
   - Buat folder untuk bot:
     ```
     mkdir ~/xauusd_trading_bot
     cd ~/xauusd_trading_bot
     ```
   - Unggah semua file dari repositori ini ke VPS (misalnya, menggunakan SCP atau FileZilla):
     ```
     scp -r /local/path/to/xauusd_trading_bot user@your_vps_ip:~/xauusd_trading_bot
     ```

5. **Instal Dependensi**:
   - Instal dependensi dari `requirements.txt`:
     ```
     pip3 install -r requirements.txt
     ```
   - **Catatan untuk TA-Lib**:
     - Jika `TA-Lib` gagal diinstal, instal library TA-Lib terlebih dahulu:
       ```
       sudo apt install libta-lib-dev -y
       pip3 install TA-Lib==0.4.24
       ```
     - Alternatifnya, unduh wheel file dari [lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/) dan instal secara manual.

6. **Siapkan File Audio (Opsional)**:
   - Jika ingin menggunakan `speech_to_text_analysis`, unggah file `fomc_audio.wav` ke direktori proyek. Jika tidak, fitur ini akan diabaikan dengan aman.

### 2. Konfigurasi
- Buka file `config.py` dengan editor seperti `nano` atau `vim`:
