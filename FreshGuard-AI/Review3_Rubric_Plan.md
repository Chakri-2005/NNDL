# FreshGuard AI — Review 3 Rubric Action Plan (50 marks)
**Team Verdura** · Kousik Sarma Lakkaraju (CB.SC.U4CSE23761) · Manikanta Sai (CB.SC.U4CSE23728) · Chakravarthy Vemula (CB.SC.U4CSE23753)

This maps every rubric line to concrete work, in the order you should actually do it (not the slide order) — because a few early decisions (dataset, chosen architecture) change what later slides can say.

---

## Do these first — everything else depends on them

### Step 0a — Lock the dataset (rubric #10, 3 marks) — **biggest risk item**
Your rubric explicitly requires an **IEEE Dataport URL**. Your current Kaggle dataset does not satisfy this on its own.

**Recommended fix — switch primary dataset to:**
- **FruitsGB: Top Indian Fruits with Quality** — IEEE DataPort, Open Access (free IEEE login, no paid subscription needed)
  DOI: `10.21227/gzkn-f379` → `https://dx.doi.org/10.21227/gzkn-f379`
  Page: `https://ieee-dataport.org/open-access/fruitsgb-top-indian-fruits-quality`
  - 12,000 images, 12 classes = 6 fruit types × {Good, Bad}: **Apple, Banana, Guava, Lime, Orange, Pomegranate**
  - 1,000 images/class, 256×256, phone-camera captured, varied angles/lighting/background
  - Already cited by 4 published papers (gives you citation material for the literature survey too)
  - **Novelty angle for slide 10:** unlike generic fruit-ID datasets (Fruits-360 etc.), FruitsGB is quality-labeled at the fruit level across 6 commercially important Indian export fruits — directly matches your "quality/freshness" framing rather than species ID.

**Alternate/backup option:**
- **Oranges Classification** — `https://ieee-dataport.org/documents/oranges-classification` — 1,900+ frames extracted from a 4K conveyor-belt video in a real packing factory. Good if you want an "industrial/real deployment" novelty angle instead, but single-fruit only.

**Action:** Don't just swap silently — on slide 10, explicitly note the change from Review 2 ("Kaggle → IEEE Dataport, per rubric requirement") and justify why FruitsGB is a better/valid substitute (or supplement) for your Apple/Banana/Orange classes. If your professor requires exact class parity with your Attention-CNN's existing 6 classes (Apple/Banana/Orange × Fresh/Rotten), map FruitsGB's Good/Bad labels onto your Fresh/Rotten labels and note that Guava/Lime/Pomegranate are extra classes you can either include (broader scope) or exclude (keep 3-fruit scope, cite the rest only as "dataset also contains").

### Step 0b — Pick ONE flagship deep learning architecture (drives #6, #7, #8, #9)
Rubric items 6–9 all assume a single architecture with real numbers. Recommendation: **make the Attention-CNN baseline your flagship**, not the CNN-GRU (which you already concluded isn't a fit for the MVP). Reasons: it's simpler to derive exact complexity/math for, and you can actually train it before this review — item 9 requires real results, not projections this time.

### Step 0c — Actually train the baseline model before this review
Rubric #9 ("Results and Discussion... achieved from the architecture") and #5 ("Best values discussed for the metric") both require **measured numbers**, not "projected." Budget time to run a real training pass on the Attention-CNN with the finalized dataset before building slides 5, 8, 9.

### Step 0d — Split the literature survey by member now (rubric #2, 5 marks)
Each student finds **5 papers**, all published in journals listed on **scimagojr.com** (SJR rankings) — this is the professor's way of blocking predatory/unindexed journals, so for each paper you must be able to show its journal on a Scimago search. Suggested split so it maps onto the architecture slides each member can then own:

| Member | 5-paper focus |
|---|---|
| Kousik Sarma Lakkaraju | Attention-CNN / CNN + attention mechanisms for image classification |
| Manikanta Sai | Hyperparameter tuning methodology (Bayesian optimization, Optuna-style search) + performance metrics for imbalanced classification |
| Chakravarthy Vemula | GRU / sequence models for visual quality tracking + on-device/edge deployment (quantization, latency) |

**Per-paper checklist:**
1. Find the paper (Google Scholar / IEEE Xplore / ScienceDirect), note the exact journal name.
2. Go to `scimagojr.com` → Journal Search → paste the journal name → confirm it's listed, note its SJR quartile (Q1–Q4) and subject category.
3. Screenshot or note the Scimago listing — professor may ask to see it.
4. Record: Title, Authors, Year, Journal, Quartile, 1–2 line relevance note, and the specific value/finding you'll cite (e.g., an accuracy number, a technique).
- 15 papers total (3 × 5), but keep 3–4 as your shared "core" citations reused across slides 2, 5, 6, and 12.

---

## Slide-by-slide plan

### 1. Problem statement, introduction, motivation — 4 marks
Reuse your Review 1/2 framing (food waste, spoilage detection, safety) but sharpen it into three distinct one-line statements: **Problem** (manual inspection is slow/subjective/inconsistent) → **Motivation** (food safety + waste reduction, on-device = accessible, no cloud dependency) → **Objective** (real-time, offline, safety-first freshness classifier). Keep to one slide, no more than 4–5 bullets.

### 2. Literature survey — 5 marks
Output: a table (Author/Year, Title, Journal + Quartile, Key contribution, Relevance to FreshGuard AI), 15 rows, grouped by the 3 members' focus areas from Step 0d. Add a line noting each journal was verified on scimagojr.com.

### 3. Architecture diagram for the overall application — 5 marks
You already have this (the lane diagram: Device Input → Pre-processing → Model → Safety Layer → Output). Reuse it, but make sure it reads as the **application**, not just the model — add where the mobile app / UI / API sits around the on-device pipeline, since "overall application" implies more than the inference path.

### 4. Module Details — 5 marks
New slide. Break the pipeline into discrete modules and describe each like a mini spec sheet:
| Module | Responsibility | Input → Output | Key tech |
|---|---|---|---|
| Capture Module | Acquire frame | Camera feed → RGB frame | Device camera API |
| Pre-processing Module | Normalize/segment | RGB frame → cleaned tensor | CLAHE, U-Net |
| Feature/Classification Module | Run the model | Tensor → class probabilities | Attention-CNN, INT8 |
| Confidence/Safety Module | Decide abstain or answer | Probabilities → verdict/abstain | OOD threshold logic |
| UI/Output Module | Present result | Verdict → screen | App front-end |

### 5. Performance metrics — 3 marks
Table with **Metric | Formula | Purpose | Why suitable for FreshGuard AI | Best value reported in literature**. You have the definitions already (Accuracy/Precision/Recall/F1) — the new column is "best value from your 15 surveyed papers" (e.g., "EfficientNetB0 OOD-aware: 98.10% acc. [cite]"), which ties this slide directly to your literature survey.

### 6. Deep learning architecture — 5 marks
For the Attention-CNN (your flagship): (a) diagram — you have this from Review 2, refine it; (b) explanation — one paragraph per layer/block; (c) **novelty** — be explicit and honest: e.g., "combines a lightweight 3-block CNN with a CBAM-style attention gate specifically tuned for spoilage-relevant regions, plus a joint confidence-abstention layer for safety-critical deployment" — the novelty is the *combination + deployment constraint*, not a brand-new layer type, and that's a legitimate novelty claim if framed that way; (d) **Time & Space complexity** — this is new work:
- **Space:** count parameters per layer (conv filters × kernel² × channels + biases), sum for total param count, note the INT8-quantized model size in MB.
- **Time:** compute FLOPs per conv layer (≈ output_H × output_W × output_C × kernel² × input_C), sum across layers, express as Big-O in terms of image size and channel depth, and relate it to your measured <30ms latency.

### 7. Algorithm Procedure — mathematical, step-by-step — 5 marks
Write the forward pass as numbered equations: input normalization → convolution (with the standard conv equation) → ReLU → pooling → attention weighting (the channel/spatial gating equations) → global average pool → dense + softmax. Follow with the **training** procedure as pseudo-code with math: loss function definition, gradient update rule (Adam equations), and the stopping criterion. This slide should look like a numbered derivation, not prose.

### 8. Hyperparameter details table with justification — 5 marks
You have the values table already — add a **Justification** column citing either your literature survey or your own tuning results (once Step 0c is done), e.g., "Dropout 0.3 — matches regularization levels reported in [cite] for similar-sized CNNs on food-image datasets."

### 9. Results and Discussion — 3 marks
Requires real numbers post-training: a results table (accuracy/precision/recall/F1 on your test split), a training curve or confusion matrix graph, and 2–3 sentences of inference — what worked, what the confusion matrix reveals (e.g., which class pairs get confused), how it compares to the literature-survey benchmarks from slide 5.

### 10. Dataset chosen — 3 marks
See Step 0a. Slide needs: dataset name, IEEE Dataport URL/DOI, class breakdown table, and an explicit "novelty" paragraph — what makes this dataset a good/novel fit versus generic alternatives.

### 11. UI screens planned — 5 marks
Actual mockups this time (not just a text description) — 3–4 screens: Capture/Camera screen, Instant Result screen (Fresh/Moderate/Spoiled + confidence), Abstention screen (safety fallback message), and optionally a History/Tracker screen if you keep the CNN-GRU as opt-in v2. I can generate these as an artifact or image set if you want — say the word.

### 12. Standard paper chosen — 2 marks
Pick one well-cited paper that most closely anchors your overall approach (attention-based lightweight CNN for food quality classification, from a Scimago-listed journal) as your "baseline paper." State its Title, why you chose it (closest match to your architecture/dataset/deployment constraints), and what you're doing differently (ties back to your novelty claim in #6).

---

## Suggested build order (not slide order)
1. Finalize dataset (0a) → 2. Train baseline for real (0c) → 3. Literature survey split & search (0d) → 4. Slides 5, 6, 7, 8, 9 (all depend on trained model + citations) → 5. Slides 3, 4, 10 (architecture/module/dataset) → 6. Slides 1, 2, 12 (framing + survey + standard paper) → 7. Slide 11 (UI mockups) → 8. Full-deck visual QA.

## Quick mark-risk summary
| Item | Marks | Risk if skipped |
|---|---|---|
| Dataset IEEE Dataport URL | 3 | Direct rubric miss — easy to lose all 3 |
| Real trained results (not projections) | Feeds #5, #9 (6 total) | Biggest single point of failure this review |
| Time/Space complexity | Part of #6 (5) | Often forgotten — do the FLOPs/param math explicitly |
| Per-student Scimago-verified papers | 5 | Needs the most lead time — start now |
