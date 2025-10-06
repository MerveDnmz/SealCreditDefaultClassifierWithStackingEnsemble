## CSE 655 – Makale Sunumu (TR Taslak)

Öğrenci: Merve DÖNMEZ (244201001016)

Sunum süresi: 15–20 dk (3 makale)

---

### 0) Giriş (1 slayt)
- Problem: Gizlilik-korumalı makine öğrenmesi; veriyi paylaşmadan modelleme.
- Çözüm ekseni: Tam homomorfik şifreleme (FHE), CKKS/SEAL ile sayısal işlemler.
- Proje bağlamı: Kredi temerrüt tahmini, CKKS ile şifreli akış + stacking.
- Sunum akışı: 3 güncel çalışma + eleştirel değerlendirme + tez bağlantısı.

---

### 1) Practical considerations of fully homomorphic encryption in privacy-preserving ML (BigData 2024)
- Problem/Motivasyon: FHE’nin ML’de pratik kullanımı; gizlilik-süre/bellek dengesi.
- Yöntem: FHE iş akışında pratik ayarlar (parametreleme, ölçek, yeniden-ölçekleme), maliyet analizi.
- Veri/Deney: Farklı görevlerde HE ile türetilmiş zaman/bellek ölçümleri, uygulama önerileri.
- Bulgular: Doğruluk çoğu senaryoda korunurken, hesaplama süresi belirgin artıyor.
- Katkılar: Mühendislik rehberi; parametre/batch/işlem adımları için pratik öneriler.
- Sınırlılıklar: Donanım bağımlılığı, bazı ağ/topolojilerde aşırı maliyet.
- Projenle bağlantı: CKKS parametre seçimi, batch_size=256 optimizasyonu, süre/bellek profilinin raporlanması.

Konuşma noktaları:
1) CKKS’te ölçek ve modül derecesi seçiminin maliyete etkisi
2) Şifreli/şifresiz akışta değerlendirme metriklerinin karşılaştırılması
3) Batching ve paralelleştirmenin pratik sınırları
4) Önerilen best-practice’lerin projeye aktarımı

---

### 2) Performance comparison of HE CNN inference between Microsoft SEAL and OpenFHE (DEIM 2023)
- Problem: Farklı HE kütüphanelerinin (SEAL vs OpenFHE) pratik performans farkları.
- Yöntem: CNN çıkarımında zaman/bellek/latency kıyaslaması; CKKS ayarları.
- Veri: MNIST/CIFAR-10 vb. (görsel) – ama yöntemsel kıyas ML genelinde fikir veriyor.
- Bulgular: Kütüphaneler arasında belirli işlemlerde belirgin performans farkları.
- Katkılar: Kütüphane seçiminin iş yüküne göre değişmesi gerektiğini gösterir.
- Sınırlılıklar: Görsel veri odaklı; tabular senaryolar doğrudan kapsanmıyor.
- Projenle bağlantı: SEAL kullanımının doğrulanması; parametre/batch etkisini anlama.

Konuşma noktaları:
1) Kütüphane mimarisi ve CKKS uygulama farkları
2) Çıkarım gecikmesinin pratik etkileri
3) Parametre setlerinin sonuçlara duyarlılığı
4) Tabular kredi verilerine aktarım: benzer maliyet desenleri

---

### 3) HEProfiler: In-depth profiler of approximate HE libraries (Preprint 2022)
- Problem: CKKS tabanlı kütüphaneler için ayrıntılı profil çıkarma ihtiyacı.
- Yöntem: İlkel işlemler (add/mul/rotate) ve bootstrapping maliyet profilinin çıkarılması.
- Bulgular: Kütüphaneler ve parametreler arasında ciddi maliyet farkları; çok-iş parçacığı etkisi.
- Katkılar: Geliştiriciler için ayrıntılı performans teşhisi aracı/bakış.
- Sınırlılıklar: Model-seviyesi uçtan uca senaryolar sınırlı olabilir.
- Projenle bağlantı: Batch ve ilkel işlem sayısının optimizasyonu; darboğazların tespiti.

Konuşma noktaları:
1) İlkel işlem maliyetleri → uçtan uca süre tahmini
2) Çoklu iş parçacığı/çekirdek kullanımının getirisi
3) Parametre ayarları ve gürültü/scale yönetimi
4) Projede ölçtüğün süre/bellek değerlerinin yorumlanması

---

### 4) Eleştirel Değerlendirme (1 slayt)
- Güçlü yönler: Gizlilik korunurken doğruluk korunabiliyor; rehberlik sağlayan güncel çalışmalar.
- Sınırlılıklar: Süre/bellek maliyetleri; belirli topolojilerde kullanılabilirlik kısıtları.
- Fırsatlar: Batch ve paralel şifreleme; GPU hızlandırma; daha iyi parametre tuning.

---

### 5) Tez/Proje Bağlantısı ve Yol Haritası (1 slayt)
- Bağlantı: CKKS/SEAL ile tabular kredi riski modelleme; stacking meta-öğrenici.
- Plan: Batch/pipeline optimizasyonu, GPU destekli booster’lar, erken durdurma.
- Çıktı: Şifreli/şifresiz AUC/süre/bellek karşılaştırması; pratik öneriler.

---

### Kaynaklar (sunum sonu)
- Lo et al., IEEE BigData 2024
- Zhu et al., DEIM 2023
- Takeshita et al., Research Square 2022


