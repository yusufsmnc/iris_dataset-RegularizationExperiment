# iris_dataset-RegularizationExperiment

# Yapay Sinir Ağlarında L1, L2 Regülarizasyon ve Dropout Etkisini İnceleme

Bu not defteri, PyTorch kullanarak Iris veri seti üzerinde eğitilen bir Yapay Sinir Ağı'nda (YSA) L1 ve L2 regülarizasyonunun yanı sıra Dropout'un model performansı üzerindeki etkisini incelemektedir.

## İçerik:

1.  **Gerekli Kütüphanelerin İçe Aktarılması**: PyTorch, NumPy, Matplotlib, scikit-learn ve Seaborn gibi temel veri analizi, modelleme ve görselleştirme kütüphaneleri yüklenir.
2.  **Veri Yükleme ve Ön İşleme**: Seaborn kütüphanesinden Iris veri seti yüklenir. Tür etiketleri sayısal değerlere dönüştürülür ve veri PyTorch tensörlerine hazırlanır.
3.  **Veri Bölme ve Yükleyiciler**: Veri, eğitim ve test setlerine ayrılır. `DataLoader` nesneleri oluşturularak verinin mini partiler halinde modele beslenmesi sağlanır.
4.  **Model Tanımı (`theModelClass`)**: Dropout katmanları içeren, 4 girişli (Iris özellik sayısı), 2 adet gizli katmana (her biri 12 nöronlu) ve 3 çıkışlı (Iris türü sayısı) basit bir feedforward YSA tanımlanır. ReLU aktivasyon fonksiyonları kullanılır.
5.  **Model Oluşturma Fonksiyonu (`createANewModel`)**: Model nesnesini, Cross-Entropy Loss fonksiyonunu, SGD optimizer'ını ve L1 regülarizasyon katsayısını (`L1lambda`) oluşturan bir yardımcı fonksiyon. Optimizer, L2 regülarizasyonunu (`weight_decay`) da içerecek şekilde yapılandırılır.
6.  **Model Eğitim Fonksiyonu (`trainTheModel`)**: Modelin eğitim döngüsünü uygular. Her epokta eğitim ve test setleri üzerinde doğruluk ve kayıp değerleri hesaplanır. L1 regülarizasyonu kayıp fonksiyonuna manuel olarak eklenir.
7.  **Tek Bir Model Eğitimi ve Sonuçların Görselleştirilmesi**: Belirli bir L1, L2 ve Dropout oranı ile tek bir model eğitilir ve eğitim/test kayıpları ile doğrulukları zamanla nasıl değiştiğini gösteren grafikler çizilir.
8.  **Düzeltme Fonksiyonu (`smooth`)**: Grafiklerdeki gürültüyü azaltmak için doğruluk ve kayıp eğrilerini düzeltmek için kullanılan bir yardımcı fonksiyon.
9.  **L2 Regülarizasyon Deneyi**: Farklı L2 regülarizasyon katsayılarının (0'dan 0.1'e kadar 10 farklı değer) modelin eğitim ve test doğruluğu üzerindeki etkisini sistematik olarak inceler. Her L2 değeri için yeni bir model eğitilir.
10. **L2 Regülarizasyon Deney Sonuçlarının Görselleştirilmesi**: Farklı L2 katsayıları için eğitim ve test doğruluklarının epoklar boyunca nasıl değiştiğini gösteren grafikler sunar.

## Amaç:

Bu not defterinin temel amacı, L1 ve L2 regülarizasyon tekniklerinin ve Dropout'un aşırı uyumu (overfitting) azaltmada ve modelin genelleme yeteneğini artırmadaki rollerini pratik bir şekilde göstermektir. Özellikle L2 regülarizasyonunun farklı seviyelerinin model performansı üzerindeki etkisi detaylı olarak incelenmiştir.
