# AI Usage Log — Semantic Resume Matcher (COMP6826001)

Log ini memenuhi Project Guideline §20 (AI Usage Log), §21 (AI and Experimental Integrity), dan §22 (AI Evaluation). Tabel di bawah akan disalin ke Technical Report bagian 10 (AI Usage Declaration).

## Aturan Penulisan

### Apa yang dicatat
- Catat setiap penggunaan AI yang **material**, yaitu yang memengaruhi kode, arsitektur, desain eksperimen, analisis, atau isi laporan. Koreksi typo atau pertanyaan umum yang tidak dipakai tidak perlu dicatat.
- Satu baris = satu tujuan penggunaan, bukan satu prompt. Satu sesi bisa menghasilkan 1–3 baris jika tujuannya berbeda (misalnya desain eksperimen dan debugging).
- Penggunaan Claude Code dicatat per fase `docs/PLAN.md`, satu baris per tujuan (misalnya "Phase 2 — coding pipeline preprocessing").
- Jangan salin percakapan. Cukup ringkasan yang menunjukkan bagaimana AI dipakai (§20).
- Aplikasi tidak memakai layanan AI eksternal. Model pretrained yang dieksperimenkan (MiniLM, mpnet) adalah bagian dari sistem DL dan didokumentasikan di bab Model dan System Design, bukan di log ini.

### Isi tiap kolom
| Kolom | Isi | Contoh |
| :--- | :--- | :--- |
| **No.** | Nomor urut, tidak pernah dipakai ulang atau diurutkan ulang. | 4 |
| **Tanggal** | Tanggal sesi, format `YYYY-MM-DD`. | 2026-10-05 |
| **AI Tool** | Nama tool + model, bukan sekadar "LLM". | Claude Code (Claude Desktop) + Colab MCP |
| **Purpose** | Kategori: brainstorming, problem definition, system design, coding, debugging, experimental design, data analysis, documentation, writing/editing, UI development. | Debugging |
| **Prompt/Instruction Summary** | Satu kalimat tentang apa yang diminta. | Asked why val P@10 drops after epoch 2 in B2 |
| **Output Used** | `Used` / `Partially used` / `Not used`, plus bagian mana yang dipakai. | Partially used — batch sampler fix only |
| **Student Verification** | Tindakan konkret yang dilakukan, hasilnya, dan keputusan akhir. Lihat aturan di bawah. | Checked group IDs per batch on 200 batches; duplicates dropped from 31% to 0%; kept fix, rejected suggested lr change |

### Aturan kolom Student Verification
- Tulis **apa yang dilakukan + hasil/bukti + keputusan**. "Checked", "reviewed", atau "looks correct" tidak cukup.
- Jika menemukan kesalahan atau keterbatasan output AI, tulis di sini. Ini yang dinilai pada rubrik *AI Usage, Verification & Reflection* (10%).
- Hanya tulis verifikasi yang **sudah benar-benar dilakukan**. Jika belum, tulis `PENDING: <rencana verifikasi>` dan perbarui setelah dilakukan. Sebelum laporan dikumpulkan, tidak boleh ada `PENDING` yang tersisa.
- Pernyataan AI tidak pernah menjadi bukti verifikasi. Metric, hasil training, label anotasi requirement, kamus sinonim final, keputusan mapping kategori, dan data user testing tidak boleh berasal dari AI (§21).

### Instruksi untuk AI saat diminta membuat entry dari suatu sesi
1. Baca sesi yang dimaksud, lalu tulis entry secara ringkas dan padat: satu kalimat per sel, dalam bahasa Indonesia.
2. Isi hanya yang terlihat jelas dari sesi. Jangan menebak tanggal, nomor urut, bagian output yang dipakai, atau verifikasi.
3. Jika ada detail yang hilang, **tanyakan langsung ke user dalam satu pesan**, biasanya:
   - nomor entry terakhir di log;
   - output mana yang dipakai, diubah, atau dibuang;
   - verifikasi apa yang sudah dilakukan beserta hasilnya (atau apakah ditandai `PENDING`).
4. Jangan pernah mengarang verifikasi, hasil tes, atau angka.

## AI Usage Log

| No. | Tanggal | AI Tool | Purpose | Prompt/Instruction Summary | Output Used | Student Verification |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-09-29 | Claude Opus 5.5 (claude.ai), dengan web search | Problem definition & experimental design | Meminta evaluasi kesesuaian ide ATS screener terhadap guideline, lalu penyusunan ulang problem statement, desain eksperimen, dan pembeda proyek. | Partially used: fine-tuning contrastive sebagai skenario wajib, desain satu track matching (TF-IDF, Siamese BiLSTM dari scratch, MiniLM frozen dan fine-tuned dengan variasi ukuran data, mpnet), robustness suite, dan evaluasi requirement gap dipakai; versi bilingual, audit bias, skenario CV tipis, cross-encoder, track klasifikasi kategori, dan fitur rewrite LLM tidak dipakai. | Membatasi scope ke CV bahasa Inggris dan dua evaluasi tambahan untuk mencegah scope creep, membuang fitur rewrite LLM, serta mempertanyakan keterkaitan dua track usulan AI sehingga desain disederhanakan menjadi satu track dengan model scratch di dalamnya; PENDING: memverifikasi klaim AI tentang dataset (tidak ada kategori Data Science, nilai kolom function/industry, batas panjang input MiniLM) saat EDA. |
| 2 | 2026-09-29 | Claude Code (Claude Desktop, Claude Opus 5.5) + Colab MCP | System design & experimental design | Milestone 1: meminta review PLAN dan CLAUDE.md terhadap guideline, usulan struktur repo, dependency, dan alur kerja git, Colab, dan Drive, lalu implementasi setup dan uji koneksi Colab. | Partially used: struktur repo, `env.py` (run folder dan `env.json`), notebook bootstrap, pembagian test lokal dan Colab, serta varian `B2-rand` untuk mengisolasi efek pretraining dipakai; rekomendasi repo private dan saran menunda keputusan `B2-rand` ke milestone 5 tidak dipakai. | Memilih repo public (berbeda dari rekomendasi AI), meminta penjelasan confound BS vs B2 sebelum mengadopsi `B2-rand`, dan mengoreksi kategori Purpose usulan AI yang melewatkan experimental design; membuka `env.json` di Drive dan nama GPU-nya cocok dengan output `nvidia-smi` (A100-SXM4-40GB); memastikan repo GitHub tidak berisi data atau `docs/reference/`; PENDING: membaca `env.py` dan `test_env.py` sampai paham fungsinya. |
