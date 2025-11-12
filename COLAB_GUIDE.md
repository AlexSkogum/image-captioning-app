# Google Colab Training Guide

## Snabb Start (5 minuter)

### 1. **Öppna Colab och ladda notebook**
   - Gå till: https://colab.research.google.com
   - Klicka **File → Open notebook**
   - Välj **GitHub** tab
   - Sök: `AlexSkogum/image-captioning-app`
   - Öppna `colab_train_flickr8k.ipynb`

### 2. **Aktivera GPU**
   - Gå till **Runtime → Change runtime type**
   - Välj **GPU** som accelerator
   - Klicka **Save**

### 3. **Kör cellerna i ordning**
   - Starta från toppen
   - Vänta på att varje cell slutförs innan nästa

### 4. **Kaggle API Setup** (VIKTIGT!)
   Du behöver ladda ner en Kaggle API-token:
   
   **Steg:**
   1. Gå till: https://www.kaggle.com/settings/account
   2. Klicka **Create New API Token**
   3. En fil `kaggle.json` laddas ner
   4. I Colab-cellen, uncommenta raden: `files.upload()`
   5. Välj `kaggle.json`-filen och ladda upp

   **ELLER:** Kopiera innehållet i `kaggle.json` och ersätt i cellen:
   ```python
   kaggle_config = {
       "username": "ditt_kaggle_användarnamn",
       "key": "din_kaggle_api_nyckel"
   }
   ```

### 5. **Efter träning**
   - Ladda ner `best.pth` checkpointen
   - Placera i `checkpoints/` på din lokala dator
   - Starta API och Gradio UI igen lokalt

---

## Tidskattning

| Steg | Tid |
|------|-----|
| Setup + download | 5 min |
| Flick8k dataset download | 3-5 min |
| Dataset prep | 2 min |
| Vocabulary build | 1 min |
| **Training** (20 epochs) | **8-12 min** |
| **Total** | **~20 min** |

---

## Felsökning

###  "CUDA not available"
→ Du är inte på GPU. Gå till **Runtime → Change runtime type → GPU**

###  "Kaggle API error"
→ Kontrollera att `kaggle.json` är rätt konfigurerad
→ Använd https://www.kaggle.com/settings/api för att generera ny token

### "Memory error"
→ Minska `batch_size` i `configs/config.yaml` (från 32 till 16)

###  "Dataset already exists"
→ Kör denna cell för att rensa:
```python
!rm -rf data/Flickr_Data/
```
Sedan kör dataset download igen.

---

## Tips

- **Spara ofta:** Colab kan timeout. Ladda ner checkpoint mellan sessioner.
- **Använda Google Drive:** Spara checkpoints på Drive för persistence:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- **Monitor GPU:** Kör denna cell för att se GPU-användning:
  ```python
  !nvidia-smi
  ```

---

## Nästa Steg

Efter träning lokalt:

1. **Kopiera `best.pth`** från Colab till `checkpoints/` lokalt
2. **Starta API:**
   ```bash
   python -m uvicorn src.api.main:app --reload --port 8000
   ```
3. **Starta Gradio:**
   ```bash
   python web/gradio_app.py
   ```
4. **Testa på:** http://localhost:7860

**Lycka till! **
