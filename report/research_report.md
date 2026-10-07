# Beyond WER: Probing Speech-Status Information in Self-Supervised Speech Representations

## Abstract

Automatic speech recognition is commonly evaluated by how accurately a
model recovers the spoken words. However, a correct transcript does not
capture many properties of speech itself, including timing, pauses,
prosody, voice characteristics, and recording-dependent acoustic
variation. These properties may be particularly relevant when analyzing
atypical or impaired speech.

This project investigates whether self-supervised speech representations
preserve information associated with dysarthric versus healthy speech
that is not captured by the text transcript alone. Experiments are
conducted on the TORGO corpus using frozen HuBERT representations,
interpretable acoustic features, Whisper transcription, and strictly
speaker-independent five-fold cross-validation.

Three increasingly controlled evaluation settings are considered: the
full TORGO dataset, utterances transcribed perfectly by Whisper, and a
stricter subset containing only perfectly transcribed prompts
represented by at least three healthy and three dysarthric speakers.
HuBERT achieved mean AUROCs of **0.836**, **0.828**, and **0.814**,
respectively. The limited decrease under increasingly strict transcript
controls indicates that speech-status-associated information remains
accessible even when lexical content and transcription errors are
substantially controlled.

Linear probing further showed that several measured acoustic
characteristics---including voiced duration, mean fundamental frequency,
pause ratio, and recording duration---are recoverable from frozen HuBERT
representations. Removing HuBERT components linearly predictable from
measured acoustic features reduced mean AUROC from **0.836 to 0.674**,
with timing- and energy-related feature groups producing the largest
individual reductions.

However, these effects varied substantially across unseen speakers and
recording conditions. With only 15 speakers, and with speaker identity
fully confounded with speech status, the results should be interpreted
as evidence about representation structure within TORGO rather than as
evidence of a clinically generalizable dysarthria classifier.

## 1. Introduction

Modern automatic speech recognition systems can produce highly accurate
text transcripts. Yet transcription captures only one component of
spoken communication: **what was said**.

The speech signal additionally contains information about **how it was
said**. Examples include speaking timing, pauses, pitch characteristics,
energy, pronunciation, voice quality, and other acoustic patterns. A
system evaluated only through transcription accuracy may therefore
discard information that remains present in the underlying speech
signal.

This distinction motivates a simple question:

> **What remains in a speech representation when the transcript is
> already correct?**

This project studies that question in the context of dysarthric and
healthy speech using the TORGO corpus. Rather than developing an
end-to-end diagnostic classifier, the objective is to investigate the
information encoded by a pretrained self-supervised speech
representation.

The central research question is:

> **Do self-supervised speech representations preserve
> dysarthria-associated information that is not captured by a correct
> text transcription?**

The analysis uses frozen HuBERT representations and simple linear models
to deliberately limit classifier complexity. This makes the experiments
closer to representation probing: if a linear classifier can separate
speech-status groups from frozen embeddings, the relevant information
must already be accessible in the representation.

A second set of experiments asks what kinds of acoustic information may
contribute to this separability and whether these effects remain
consistent across speakers and recording conditions.

## 2. Research Questions

### RQ1 --- Beyond transcription accuracy

> Does speech-status-associated separability remain when Whisper
> produces a perfect transcription?

### RQ2 --- Controlling lexical content

> Does separability remain when analysis is additionally restricted to
> prompts shared across multiple healthy and dysarthric speakers?

### RQ3 --- Acoustic information in HuBERT

> Which interpretable acoustic characteristics are linearly accessible
> from frozen HuBERT representations?

### RQ4 --- Acoustic contribution and speaker dependence

> How does speech-status separability change when components of the
> HuBERT representation that are linearly predictable from measured
> acoustic characteristics are removed?

## 3. Dataset

### 3.1 TORGO

Experiments use the **TORGO dysarthric speech corpus**.

  Property                    Value
  ----------------------- ---------
  Recordings                 16,552
  Healthy recordings         10,978
  Dysarthric recordings       5,574
  Speakers                       15
  Healthy speakers                7
  Dysarthric speakers             8
  Total audio duration      13.68 h
  Unique transcriptions         960

The dataset contains repeated prompts spoken by multiple speakers, which
enables comparisons under shared lexical content.

An important limitation is that each speaker belongs exclusively to one
speech-status group. Consequently, **speaker identity and speech status
are fully confounded**. A model may therefore exploit speaker-specific
or recording-specific characteristics that correlate with the target
label.

For this reason, all primary experiments use speaker-independent
evaluation.

### 3.2 Speaker-independent cross-validation

A fixed five-fold speaker-independent split was created.

  Fold   Dysarthric test speakers   Healthy test speakers
  ------ -------------------------- -----------------------
  1      F01, M02                   MC01
  2      F03, M05                   MC03
  3      M01, M04                   FC03
  4      M03                        FC01, FC02
  5      F04                        MC02, MC04

No speaker appears in both the training and test partition of a fold.
The same folds are reused throughout the project.

## 4. Representations

### 4.1 Interpretable acoustic features

Ten acoustic features were extracted from each recording:

-   audio duration
-   voiced duration
-   speech ratio
-   pause ratio
-   number of pauses
-   mean pause duration
-   mean fundamental frequency (F0)
-   F0 variability
-   mean RMS energy
-   RMS variability

Non-silent regions were estimated using energy-based segmentation.
Pause-related quantities should therefore be interpreted as
**energy-derived timing measures**, not linguistically annotated pauses
or output from a learned voice-activity detector. RMS measurements can
also be affected by microphone gain and recording conditions.

### 4.2 HuBERT representations

Self-supervised representations were extracted using pretrained **HuBERT
Base (`facebook/hubert-base-ls960`)**. The encoder remained completely
frozen.

Frame-level hidden representations were temporally mean-pooled into a
single **768-dimensional embedding**. Padding was excluded from pooling
through the corresponding attention mask. Recordings longer than 30
seconds were processed in non-overlapping 30-second chunks and
aggregated using duration-weighted mean pooling.

A balanced logistic-regression classifier was trained on the frozen
embeddings. The model therefore functions as a **linear probe** rather
than a fine-tuned speech classifier.

## 5. Beyond-WER Experimental Design

Three increasingly restrictive evaluation conditions were constructed.

### 5.1 Full TORGO

All **16,552 recordings**.

### 5.2 Perfect ASR

Whisper Small was used to transcribe the recordings. The subset contains
only recordings for which normalized Whisper output exactly matches the
normalized TORGO reference at word level (**WER = 0**):

-   **10,121 recordings**
-   8,104 healthy
-   2,017 dysarthric
-   all 15 speakers retained

### 5.3 Strict shared-prompt Perfect ASR

A prompt was retained only if Whisper transcribed it perfectly and at
least **three unique healthy speakers** and **three unique dysarthric
speakers** produced that normalized TORGO prompt.

The resulting subset contains:

-   **4,962 recordings**
-   **213 shared prompts**
-   3,402 healthy
-   1,560 dysarthric
-   all 15 speakers

This reduces---but does not eliminate---content and speaker-related
confounding. It is not a one-to-one matched-pairs design.

## 6. Beyond-WER Results

  ------------------------------------------------------------------------------------
  Representation   Condition                  N     Balanced     Macro F1        AUROC
                                                    Accuracy              
  ---------------- --------------- ------------ ------------ ------------ ------------
  Acoustic         Full TORGO            16,552  .693 ± .117  .652 ± .136  .771 ± .114

  Acoustic         Perfect ASR           10,121  .687 ± .086  .604 ± .086  .761 ± .114

  Acoustic         Strict                 4,962  .687 ± .104  .636 ± .121  .778 ± .101
                   shared-prompt                                          

  HuBERT           Full TORGO            16,552     **.723 ±     **.694 ±     **.836 ±
                                                      .136**       .149**       .165**

  HuBERT           Perfect ASR           10,121     **.723 ±     **.656 ±     **.828 ±
                                                      .131**       .157**       .161**

  HuBERT           Strict                 4,962     **.709 ±     **.668 ±     **.814 ±
                   shared-prompt                      .148**       .164**       .179**
  ------------------------------------------------------------------------------------

![Speech-status separability under transcript
controls](../results/figures/figure_1_beyond_wer.png)

HuBERT AUROC changes from **0.836 → 0.828 → 0.814**, while balanced
accuracy changes from **0.723 → 0.723 → 0.709**. Performance therefore
does not collapse after selecting only perfectly transcribed recordings
or additionally restricting analysis to prompts shared across multiple
speakers from both groups.

Within TORGO, this supports the hypothesis that speech representations
contain speech-status-associated information beyond what is captured by
a correct transcription. It does **not** establish clinically
generalizable dysarthria detection.

## 7. Probing Acoustic Information in HuBERT

Linear Ridge regression probes were trained to predict individual
acoustic measurements from frozen HuBERT embeddings using the same
speaker-independent folds.

  Acoustic characteristic              Mean R²
  ------------------------- ------------------
  Voiced duration              **.578 ± .160**
  Mean F0                      **.515 ± .097**
  Pause ratio                  **.483 ± .076**
  Audio duration               **.474 ± .160**
  RMS mean                     **.367 ± .191**
  F0 variability               **.301 ± .169**
  RMS variability              **.281 ± .269**
  Mean pause duration          **.112 ± .135**
  Number of pauses            **−.305 ± .642**

![Acoustic characteristics accessible from
HuBERT](../results/figures/figure_4_acoustic_probe.png)

Several acoustic properties are substantially linearly predictable from
frozen HuBERT embeddings across unseen speakers. The negative mean R²
for number of pauses indicates poor cross-speaker generalization for
this target.

These results demonstrate **linear accessibility**, not that HuBERT
necessarily uses these characteristics to separate speech-status groups.

## 8. Acoustic Residualization

Within each fold, a Ridge regression model learned a mapping from
selected acoustic measurements to the standardized HuBERT representation
using training speakers only.

For HuBERT embedding (H) and acoustic vector (A):

\[ H\_{`\mathrm{residual}`{=tex}} = H - `\hat{H}`{=tex}(A) \]

The residual therefore removes only the HuBERT component **linearly
predictable from the selected measured features**.

### 8.1 Feature groups

-   **Duration:** audio duration
-   **Timing:** voiced duration, pause ratio, number of pauses, mean
    pause duration
-   **Prosody:** mean F0, F0 variability
-   **Energy:** RMS mean, RMS variability
-   **All measured acoustics:** all features above

Speech ratio was excluded from the combined feature set because it is
redundant with pause ratio.

### 8.2 Results

  Representation                                     AUROC     Δ AUROC
  -------------------------------------- ----------------- -----------
  Original HuBERT                          **.836 ± .165**         ---
  Residualized: duration                       .811 ± .179       −.024
  Residualized: prosody                        .809 ± .163       −.027
  Residualized: timing                         .737 ± .255       −.099
  Residualized: energy                         .740 ± .252       −.096
  Residualized: all measured acoustics     **.674 ± .261**   **−.162**

![HuBERT separability after acoustic
residualization](../results/figures/figure_2_residual_ablation.png)

Timing and energy produce substantially larger reductions than duration
and prosody. Joint residualization produces the largest reduction, from
**0.836 to 0.674 AUROC**.

This indicates overlap between speech-status-associated HuBERT structure
and representation structure predictable from the measured acoustic
characteristics. The result is not causal: residualization can alter
correlated dimensions and does not remove all acoustic information.

## 9. Speaker-Dependent Effects

Out-of-fold predictions were generated for every recording. For each
speaker:

\[ `\Delta `{=tex}P =
P\_{`\mathrm{residual}`{=tex}}(`\mathrm{dysarthria}`{=tex}) -
P\_{`\mathrm{HuBERT}`{=tex}}(`\mathrm{dysarthria}`{=tex}) \]

![Speaker-dependent residualization
effects](../results/figures/figure_3_speaker_residualization_effects.png)

Examples include:

-   F04: **ΔP = −.401**
-   M05: **ΔP = −.274**
-   FC03: **ΔP = −.140**
-   F03: **ΔP = +.211**
-   MC03: **ΔP = +.249**
-   FC01: **ΔP = +.295**

F03 and F04 are both dysarthric speakers, yet residualization shifts
their predictions strongly in opposite directions.

The appropriate conclusion is:

> **Removing the HuBERT component linearly predictable from the selected
> acoustic features substantially changes speech-status predictions, but
> the direction and magnitude of this effect vary strongly across unseen
> speakers.**

## 10. Qualitative Error Analysis

Selected recordings showing large prediction changes were inspected
manually as a qualitative diagnostic.

Some F03 recordings contained noticeable background noise or relatively
quiet speech. Selected F04 recordings sounded comparatively unobtrusive
despite their TORGO dysarthria label. Several FC01 recordings were
quiet, while selected FC03 recordings generally sounded clear apart from
occasional microphone noise.

These observations were **not used to modify labels or exclude
recordings**. They served only as hypotheses motivating a systematic
recording-condition analysis.

## 11. Recording and Microphone Conditions

Two identifiable microphone conditions were analyzed: **array
microphone** and **head microphone**.

![Performance across folds and microphone
conditions](../results/figures/figure_5_microphone_stratification.png)

Across the complete out-of-fold dataset:

  Microphone           Original HuBERT AUROC   Residualized AUROC
  ------------------ ----------------------- --------------------
  Array microphone                      .779                 .689
  Head microphone                       .834                 .730

Fold-level results are substantially more heterogeneous. In Fold 5,
original AUROC decreases to approximately **.590** for array microphone
and **.702** for head microphone. After residualization, the
corresponding AUROCs fall to approximately **.260** and **.197**.

This argues against a simple explanation in which one microphone type
alone causes the observed generalization failures. The results are more
consistent with interactions among **speaker identity, recording
conditions, acoustic characteristics, and fold composition**.

## 12. Discussion

### 12.1 Speech contains information beyond its transcript

Speech-status-associated separability persists after increasingly strict
transcript controls. HuBERT achieves an AUROC of **.836** on the full
dataset and **.814** on the strict shared-prompt Perfect-ASR subset.

> **A correct transcript does not exhaust the information available in
> the speech signal.**

Within TORGO, frozen self-supervised speech representations preserve
information associated with speech status even when the lexical
transcription is correct.

### 12.2 HuBERT represents interpretable acoustic structure

Voiced duration, mean F0, pause ratio, and total duration can be
predicted to a meaningful degree from frozen HuBERT embeddings across
unseen speakers. Not every handcrafted feature is equally recoverable;
pause count generalizes poorly.

### 12.3 Measured acoustics explain part---but not all---of the separability

Joint residualization decreases AUROC from **.836 to .674**, while
residual representations retain some speech-status-associated
separability on average. The original HuBERT result therefore cannot be
reduced to a single measured acoustic characteristic.

### 12.4 Generalization is strongly speaker dependent

Performance varies substantially across speaker-independent folds and
individual speakers. Because the dataset contains only **15 speakers**,
with speech status constant within each speaker, HuBERT may encode
speaker identity, recording environment, microphone characteristics, and
other speaker-correlated factors alongside dysarthria-associated speech
characteristics.

The project therefore demonstrates representation separability within
TORGO, not robust clinical generalization.

## 13. Limitations

### Small number of speakers

The 16,552 recordings originate from only 15 speakers. Repeated
recordings from the same speaker are not independent observations.

### Speaker-label confounding

Every speaker belongs exclusively to one speech-status group.
Speaker-independent folds reduce direct leakage but cannot eliminate
dataset-level speaker confounding.

### Recording-condition confounding

Microphone, gain, background noise, session conditions, and other
recording characteristics may contribute to learned representations. RMS
measures are particularly sensitive to these factors.

### Energy-based pause estimation

Pause-related features are derived from energy-based non-silent
segmentation rather than manually annotated pauses or a dedicated
speech-activity model.

### Linear probes

The experiments test whether information is **linearly accessible**, not
the total information contained in the representation.

### Residualization is not causal

Removing linearly predictable representation components does not
establish that the corresponding acoustic characteristics causally
determine predictions.

### Dataset-specific conclusions

No external dysarthria corpus was used for validation. Findings should
not be generalized directly to clinical populations, languages,
recording environments, or speech disorders outside TORGO.

## 14. Conclusion

This project investigated whether self-supervised speech representations
preserve information associated with dysarthric versus healthy speech
beyond what is contained in a correct text transcription.

Three results stand out:

1.  **Speech-status separability persists under increasingly strict
    transcript controls.** Frozen HuBERT representations achieve AUROCs
    of **.836**, **.828**, and **.814** on the full, Perfect-ASR, and
    strict shared-prompt conditions.
2.  **Several interpretable acoustic characteristics are linearly
    accessible from HuBERT**, particularly voiced duration, mean F0,
    pause ratio, and recording duration.
3.  **Removing HuBERT components linearly predictable from measured
    acoustic characteristics reduces speech-status separability**,
    particularly for timing- and energy-related feature groups. Joint
    residualization reduces mean AUROC from **.836 to .674**.

However, these effects are strongly speaker dependent, and
microphone-stratified analysis does not support a simple
single-recording-condition explanation.

> **Within TORGO, frozen HuBERT representations retain substantial
> speech-status-associated information after controlling transcription
> and lexical content. Part of this information overlaps with measurable
> acoustic structure, but its expression and generalization vary
> considerably across unseen speakers.**

These findings illustrate why speech assessment should be considered
**beyond WER**: accurate recognition of *what was said* does not imply
that the remaining information about *how it was said* has disappeared.

## Repository Structure

``` text
neurospeech/
├── data/
├── modal/
├── results/
│   ├── figures/
│   │   ├── figure_1_beyond_wer.png
│   │   ├── figure_2_residual_ablation.png
│   │   ├── figure_3_speaker_residualization_effects.png
│   │   ├── figure_4_acoustic_probe.png
│   │   └── figure_5_microphone_stratification.png
│   └── ...
├── scripts/
├── src/
│   └── neurospeech/
│       ├── evaluation/
│       └── features/
└── report/
    └── research_report.md
```
