# Cervical Spine MRI 3D Interactive Viewer

Interaktywna przeglądarka tomografii rezonansu magnetycznego (MRI) odcinka szyjnego kręgosłupa z rekonstrukcją wolumetryczną i modelem 3D.

## Funkcje
- **Przekroje 2D**: Przeglądanie warstw sagitalnych i osiowych w wysokiej rozdzielczości (kontrast, wzmocnienie struktur, numery kręgów C2–C7).
- **Model 3D**: Segmentacja kręgów C1–T1, krążków międzykręgowych, rdzenia kręgowego i schematu korzeni nerwowych (Three.js).
- **Objętość 3D**: Wolumetryczny raymarching chmury danych MRI T2 z wektorowymi obrysami sylwetkowymi (Anti-Aliasing) i suwakiem intensywności modelu.
- **Autoryzacja kliniczna**: Dostęp do badania zabezpieczony hasłem.

## Uruchomienie lokalne
Uruchom skrypt `uruchom_przegladarke.cmd` lub w terminalu:
```bash
python -m http.server 8765 --directory mri_viewer
```
Następnie przejdź pod adres: `http://127.0.0.1:8765/`.
