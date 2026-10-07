# NeuroSpeech: Dysarthric Speech Analysis Beyond Transcription

## Project Overview

NeuroSpeech is a research project focused on analyzing dysarthric speech with a central question: what clinically meaningful information is lost when speech is reduced to text alone? The project explores acoustic patterns, prosodic cues, and self-supervised speech representations to better understand dysarthria beyond standard automatic speech recognition (ASR) outputs.

## Research Question

What clinically relevant information about dysarthric speech is lost when speech is reduced to text?

This project addresses a key limitation of conventional ASR systems: they focus on the transcription of spoken content, but dysarthria often affects how speech is produced, not just what is said. Acoustic quality, articulation, rhythm, and vocal characteristics can carry important clinical information.

---

## Disease Under Investigation: Dysarthria

Dysarthria is a motor speech disorder caused by neurological impairment affecting the muscles involved in speech production. It is often associated with conditions such as cerebral palsy, Parkinson's disease, stroke, ALS, multiple sclerosis, and other neurological disorders.

### Key Characteristics

- Reduced speech clarity and articulation
- Slower or irregular speech rate
- Abnormal prosody, pitch, or loudness
- Breathy, harsh, or strained voice quality
- Nasalization or breathy resonance changes
- Reduced intelligibility for listeners

### Why It Matters

For dysarthric speech, the acoustic signal often contains crucial information that is not captured by a raw text transcript. Important clinical signals may lie in:

- prosody and rhythm
- articulation precision
- vocal quality
- speech timing and variation

This is why the project investigates speech representations beyond transcription and evaluates whether non-textual cues improve understanding and downstream analysis of dysarthric speech.

---

## Models Used

### 1. Whisper

Whisper is an ASR model developed by OpenAI and used as the transcription baseline in this project.

- Model: `openai/whisper-small`
- Purpose: generate text transcriptions from dysarthric audio
- Evaluation metrics:
  - WER (Word Error Rate)
  - CER (Character Error Rate)

Whisper is used to study how well standard ASR performs on dysarthric speech and where it fails.

### 2. HuBERT

HuBERT is a self-supervised speech representation model that learns meaningful speech features without relying on human-labeled transcripts for the representation task.

- Purpose: extract rich speech embeddings from raw audio
- Benefits: captures acoustic and prosodic information beyond text
- Use in this project: compare speech representations with transcription-based baselines and study what information remains useful for dysarthric speech analysis

### 3. Acoustic Feature Extraction

Classical signal-processing features are also used to characterize speech patterns such as:

- spectral structure
- MFCC-like features
- temporal dynamics
- prosodic descriptors

These features are useful for interpretability and can help isolate which speech attributes are most relevant for dysarthria-related analysis.

---

## Repository Structure

```text
Neurospeech/
├── scripts/
│   ├── audit_torgo.py
│   ├── build_metadata.py
│   ├── create_folds.py
│   ├── extract_acoustic_features.py
│   ├── inspect_torgo.py
│   ├── test_whisper.py
│   ├── test_hubert.py
│   ├── train_acoustic_baseline.py
│   ├── train_hubert_baseline.py
│   ├── evaluate_whisper_subset.py
│   ├── analyze_whisper_full.py
│   ├── analyze_hubert_acoustic_relationship.py
│   ├── analyze_prediction_errors.py
│   ├── analyze_microphone_stratified_performance.py
│   └── ...
├── src/
│   └── neurospeech/
│       ├── evaluation/
│       └── features/
├── data/
├── results/
├── report/
├── modal/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

The project uses the TORGO database, a widely used dataset for dysarthric speech research.

### TORGO Characteristics

- healthy and dysarthric speakers
- both male and female speakers
- audio recordings with associated transcriptions
- metadata for speech condition and demographics

The dataset is used for exploratory analysis, auditing, feature extraction, model evaluation, and ablation studies.

---

## Current Progress

- [x] TORGO dataset integration
- [x] Dataset inspection
- [x] Dataset audit
- [ ] Speaker-independent train/validation/test split
- [ ] Acoustic feature extraction
- [ ] ASR baseline
- [ ] Self-supervised speech representations
- [ ] Ablation experiments
- [ ] Final evaluation

---

## Project Goals

This project aims to:

1. evaluate how well standard ASR models perform on dysarthric speech
2. determine which clinically relevant speech cues are lost in text-only representations
3. compare transcription-based methods with acoustic and self-supervised speech features
4. build a foundation for more clinically meaningful speech analysis tools

---

## Dependencies

The repository uses Python libraries for ML, audio processing, and evaluation, including:

- PyTorch
- torchaudio
- Transformers
- Datasets
- librosa
- soundfile
- NumPy
- pandas
- scikit-learn
- jiwer
- matplotlib
- tqdm

See `requirements.txt` for the full dependency list.

---

## Why This Project Matters

Dysarthria is not just a transcription problem. It is a speech production disorder with distinctive acoustic patterns. By moving beyond text-only analysis, this project explores whether machine learning systems can capture clinically meaningful differences that are otherwise hidden in ordinary ASR pipelines.

This work is intended as a step toward better understanding, diagnosis, and analysis of dysarthric speech in research and clinical settings.
