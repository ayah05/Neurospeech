# NeuroSpeech: Dysarthric Speech Analysis Beyond Transcription

## Projektübersicht

NeuroSpeech ist ein Forschungsprojekt, das sich mit der Analyse dysarthrnischer Sprache (Dysarthrie) befasst. Das Projekt untersucht, welche klinisch relevanten Informationen verloren gehen, wenn Sprache nur auf Text reduziert wird. Durch die Verwendung moderner Machine-Learning-Techniken werden akustische Merkmale und selbstüberwachte Sprachrepräsentationen analysiert, um ein tieferes Verständnis dieser Sprachstörung zu erreichen.

### Forschungsfrage

**Welche klinisch relevanten Informationen über dysarthrnische Sprache gehen verloren, wenn Sprache nur auf Text reduziert wird?**

Dieses Projekt adressiert eine kritische Lücke in der Spracherkennungsforschung: Automatische Spracherkennungssysteme (ASR) konzentrieren sich ausschließlich auf Transkriptionen, ignorieren aber wichtige akustische und prosodische Merkmale, die gerade bei pathologischen Sprachen entscheidend sind.

---

## Dysarthrie: Die untersuchte Krankheit

### Was ist Dysarthrie?

**Dysarthrie** ist eine neurologische Sprachstörung, die durch eine Schwäche oder fehlende Kontrolle der Sprechmuskeln gekennzeichnet ist. Sie resultiert aus neurologischen Schädigungen und beeinträchtigt die Fähigkeit, Sprache zu produzieren.

### Hauptmerkmale von Dysarthrie

- **Articulation (Aussprache)**: Undeutliche oder verwischte Konsonanten und Vokale
- **Prosodie**: Abnormale Tonhöhe, Lautstärke und Sprechgeschwindigkeit
- **Stimme**: Heiserkeit, Rauheit oder nasale Qualität
- **Rhythmus**: Unregelmäßige Sprechgeschwindigkeit oder Pausen
- **Verständlichkeit**: Reduzierte oder eingeschränkte Verständlichkeit für Hörer

### Ursachen

Dysarthrie kann durch verschiedene neurologische Erkrankungen verursacht werden:
- **Zerebralparese** (Cerebral Palsy)
- **Parkinson-Krankheit**
- **Multiple Sklerose**
- **Schlaganfälle**
- **Hirn- oder Rückenmarksverletzungen**
- **Amyotrophe Lateralsklerose (ALS)**

### Klinische Bedeutung

Die traditionelle Spracherkennung konzentriert sich ausschließlich auf die Transkription, d.h. auf das *Was wird gesagt*, nicht auf das *Wie wird es gesagt*. Bei Dysarthrie sind jedoch die akustischen Eigenschaften (Prosodie, Articulation, Stimmqualität) oft informativer als die reinen Wörter. Diese Informationen sind unverzichtbar für:

- **Klinische Diagnostik**: Früherkennung und Verlaufskontrolle
- **Therapieplanung**: Maßgeschneiderte Interventionen
- **Prognose**: Vorhersage des Krankheitsverlaufs

---

## Modelle und Technologien

Das Projekt nutzt mehrere state-of-the-art Modelle für die Sprachanalyse:

### 1. **Whisper (OpenAI)**

**Rolle**: Automatische Spracherkennung (ASR) Baseline

Whisper ist ein großes Spracherkennungsmodell von OpenAI, trainiert auf 680.000 Stunden mehrsprachiger und mehrsprachiger Audio-Daten aus dem Internet.

- **Modell**: `openai/whisper-small`
- **Funktion**: Konvertiert Sprache in Text
- **Zweck**: Ermittelt die Baseline-Transkriptionsgenauigkeit für dysarthrnische Sprache
- **Metriken**:
  - **WER** (Word Error Rate): Prozentsatz der falsch erkannten Wörter
  - **CER** (Character Error Rate): Prozentsatz der falsch erkannten Zeichen

### 2. **HuBERT (Self-Supervised Learning)**

**Rolle**: Selbstüberwachte Sprachrepräsentation

HuBERT (Hidden-Unit BERT) ist ein selbstüberwachtes Lernmodell, das ohne manuell annotierte Daten hochdimensionale Sprachrepräsentationen lernt.

- **Architektur**: Transformer-basierte Embeddings
- **Trainingsdaten**: Nicht-gekennzeichnete Audiodaten
- **Funktion**: Extrahiert tiefe akustische Merkmale aus Rohaudio
- **Vorteil**: Erfasst prosodische und akustische Charakteristiken, die über Text-Transkriptionen hinausgehen
- **Output**: Dense Vektoren (Hidden-Unit-Embeddings), die semantische und akustische Informationen kodieren

### 3. **Acoustic Features (Klassische Merkmale)**

**Rolle**: Interpretierbare akustische Analysen

- **Merkmale**: Spektrale Eigenschaften (MFCC, Mel-Frequenz-Spektrogram), Prosodische Merkmale
- **Funktion**: Liefert traditionelle sprachverarbeitungsbasierte Features
- **Vorteil**: Direkt interpretierbar für klinische Anwendungen

---

## Projektstruktur

```
Neurospeech/
├── scripts/                              # Hauptverarbeitungsskripte
│   ├── download_torgo.py                # TORGO-Datensatz herunterladen
│   ├── inspect_torgo.py                 # Datensatz inspizieren
│   ├── audit_torgo.py                   # Datensatzaudit und Analyse
│   ├── build_metadata.py                # Metadaten erstellen
│   ├── create_folds.py                  # Train/Val/Test-Split
│   ├── extract_acoustic_features.py     # Akustische Merkmale extrahieren
│   ├── test_whisper.py                  # Whisper-Basis-Tests
│   ├── test_hubert.py                   # HuBERT-Basis-Tests
│   ├── train_acoustic_baseline.py       # Akustisches Baseline-Modell trainieren
│   ├── train_hubert_baseline.py         # HuBERT-Baseline trainieren
│   ├── evaluate_whisper_subset.py       # Whisper-Evaluierung
│   ├── analyze_whisper_full.py          # Umfassende Whisper-Analyse
│   ├── analyze_hubert_acoustic_relationship.py  # Beziehung HuBERT-Akustik
│   ├── analyze_prediction_errors.py     # Fehleranalyse
│   ├── analyze_microphone_stratified_performance.py  # Mikrofon-abhängige Performance
│   └── ...                              # Weitere Analyse- und Ablations-Skripte
├── src/neurospeech/                     # Kernmodule
│   ├── features/                        # Merkmal-Extraction
│   │   └── ssl.py                       # Self-Supervised Learning (HuBERT)
│   └── evaluation/                      # Evaluierungs-Tools
│       └── asr.py                       # ASR-Metriken (WER, CER)
├── data/                                # Datensätze und Metadaten
├── results/                             # Ergebnisse und Visualisierungen
├── report/                              # Technische Berichte
├── modal/                               # Cloud-Computing-Konfiguration
└── requirements.txt                     # Python-Abhängigkeiten
```

---

## Workflow und Fortschritt

Das Projekt folgt einem strukturierten Workflow:

### ✅ Abgeschlossene Phasen

- [x] **TORGO-Datensatz Integration**: Laden und Konfiguration des Dysarthrie-Datensatzes
- [x] **Dataset-Inspektion**: Grundlegende Analyse der Datenstruktur und Qualität
- [x] **Dataset-Audit**: Umfassende Validierung (Sprecher, Geschlecht, Dauer, Transkriptionen)

### 🔄 Laufende / Bevorstehende Phasen

- [ ] **Speaker-unabhängiger Train/Val/Test-Split**: Stratifizierte Aufteilung nach Sprechern
- [ ] **Akustische Merkmal-Extraction**: Berechnung von MFCC, Mel-Spektrogrammen, etc.
- [ ] **ASR Baseline**: Evaluierung von Whisper auf dysarthrnischen Daten
- [ ] **Self-Supervised Speech Representations**: Anwendung von HuBERT
- [ ] **Ablations-Experimente**: Systematische Analyse der Beiträge verschiedener Merkmale
- [ ] **Evaluierung & Interpretation**: Finale Ergebnisse und klinische Implikationen

---

## Datensatz: TORGO

Das Projekt verwendet den **TORGO-Datensatz** (Toronto Dysarthria Database), einen Standard-Benchmark für Dysarthrie-Forschung.

### TORGO-Eigenschaften

- **Sprecher**: Personen mit und ohne Dysarthrie
- **Geschlecht**: Männlich und weiblich
- **Transkriptionen**: Standardisierte Sätze und Wörter
- **Audiosignale**: Hochwertige Sprachaufnahmen
- **Metadaten**: Sprecherdemografien, Sprachstatus (gesund vs. Dysarthrie)

### Datensatz-Zugriff

```python
from datasets import load_dataset
dataset = load_dataset("abnerh/TORGO-database")
```

---

## Dependencies

Das Projekt nutzt moderne Python-Bibliotheken:

- **Machine Learning**: `torch`, `transformers`, `torchaudio`
- **Hugging Face Ecosystem**: `datasets`, `huggingface-hub`
- **Audio Processing**: `librosa`, `soundfile`
- **Data Analysis**: `pandas`, `numpy`, `scikit-learn`
- **Evaluation**: `jiwer` (Word Error Rate)
- **Visualization**: `matplotlib`

Siehe `requirements.txt` für die vollständige Liste.

---

## Forschungsbeiträge

Dieses Projekt trägt zu mehreren wichtigen Fragen bei:

1. **Grenzen der textbasierten ASR**: Zeigt, dass Transkriptionen allein für dysarthrnische Sprache unzureichend sind
2. **Akustische Marker**: Identifiziert aussagekräftige akustische Merkmale, die klinisch relevantem
3. **Model-Vergleiche**: Evaluiert verschiedene Ansätze (klassische Features vs. self-supervised Learning)
4. **Klinische Anwendbarkeit**: Bereitet den Weg für bessere diagnostische Tools

---

## Verwendung

### Voraussetzungen

```bash
pip install -r requirements.txt
```

### Grundlegende Schritte

1. **Datensatz inspizieren**:
   ```bash
   python scripts/inspect_torgo.py
   ```

2. **Datensatz auditieren**:
   ```bash
   python scripts/audit_torgo.py
   ```

3. **Akustische Merkmale extrahieren**:
   ```bash
   python scripts/extract_acoustic_features.py
   ```

4. **Whisper testen**:
   ```bash
   python scripts/test_whisper.py
   ```

5. **HuBERT testen**:
   ```bash
   python scripts/test_hubert.py
   ```

---

## Zitate und Referenzen

- **TORGO Dataset**: Rudzicz et al. (2012) - Toronto Dysarthria Database
- **Whisper**: Radford et al. (2022) - "Robust Speech Recognition via Large-Scale Weak Supervision"
- **HuBERT**: Hsu et al. (2021) - "HuBERT: Self-supervised Speech Representation Learning by Masked Prediction of Hidden Units"

---

## Lizenz

Dieses Projekt ist zur Forschung verfügbar. Bitte beachten Sie die Lizenzbedingungen des TORGO-Datensatzes.

---

## Kontakt & Beiträge

Für Fragen, Bug-Reports oder Beiträge: Öffnen Sie ein GitHub Issue oder erstellen Sie einen Pull Request.

**Autor**: ayah05
**Repository**: https://github.com/ayah05/Neurospeech
