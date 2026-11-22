# TinyCD demosunu Mac’te açma

Repo kökünden tek komutla demoyu başlat ve tarayıcıda **http://127.0.0.1:8501** adresini aç.

```sh
./run_tinycd_demo.sh
```

Terminali sunum boyunca açık tut. Kapatmak için terminalde **Ctrl+C** kullan.

- Epoch 40 ağırlığı `artifacts/tinycd/epoch_40-cae2a604.pt` konumunda. TinyCD’nin sabitlenmiş model kaynakları aynı klasörün altındaki `tinycd-1696827ffb7473c448591aaaa4434d4d4934a1ba/models/` dizininde. Bu dosyalar hazır olduğunda demo çevrimdışı açılır.
- Görüntüler `data/levir-cd/val/A/` ve `data/levir-cd/val/B/` altında. Referans etiketleri `data/levir-cd/val/label/` altında.
- Streamlit kurulumu eksikse mevcut proje ortamından ayrı paket dizinine kur.

  ```sh
  .venv/bin/python -m pip install --target artifacts/tinycd/streamlit-deps \
    -c requirements.txt -r requirements-demo.txt
  ```

## Sunum sırasında

1. Varsayılan **val_56** sahnesinde **Değişimi göster** düğmesine bas. Sonraki görüntüdeki mavi katman modelin tahminidir.
2. **Referansla karşılaştır** anahtarını aç. Yeşil doğru yakalanan, mor kaçırılan, turuncu fazladan işaretlenen bölgeyi gösterir. Görselin üzerindeki tam ekran düğmesiyle büyüt.
3. **Örnek seç** listesinden **val_39**’a geç. Yeniden **Değişimi göster** de ve referans görünümünde mor alanlardan birini göster.
4. Altındaki iki düğmeden 1024×1024 siyah-beyaz maskeyi veya renk açıklaması eklenmiş karşılaştırma PNG’sini indir. Ekrandaki sahne F1’i seçili örneğe, **%90,47** ise 128 çiftlik resmî LEVIR-CD testine aittir.

Canlı gösterim kesilirse `artifacts/tinycd/stage4/val_56_karsilastirma.png` ve `artifacts/tinycd/stage4/val_39_karsilastirma.png` dosyalarını doğrudan açabilirsin.
