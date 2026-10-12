# 🍏 FreshGuard AI – Project Review Preparation Document
**Slogan:** Smarter freshness for healthier living.

---

## 1. PROJECT IN ONE PAGE

**What is FreshGuard AI?**
A deep learning-based web application that detects whether a given fruit is fresh or rotten from an image.

**What problem does it solve?**
Food waste and food safety. Manual sorting is slow, error-prone, and labor-intensive. Automated visual inspection helps consumers and supply chains quickly identify spoiled produce.

**Why did we build it?**
To compare modern computer vision architectures (CNNs vs. Transformers) on a practical, real-world agricultural problem while ensuring strict data hygiene (preventing data leakage) and providing a user-friendly interface.

**What is the input?**
A 224 × 224 RGB image of an Apple, Banana, or Orange.

**What is the output?**
The fruit type, freshness state (Fresh/Rotten), and a confidence probability (e.g., "Apple • Rotten" with 98% confidence).

**Who can use it?**
Supermarket staff, agricultural supply chains, and everyday consumers.

**Overall Workflow:**
`Image Upload → Resize & Normalize → Deep Learning Backbone → Feature Extraction → Classification Head → Softmax Probability → Confidence Check → Web Dashboard Display`

---

## 2. DATASET

**Dataset Used:** Top Indian Fruits Dataset (Subset of the 12,000 clean images)
**Classes (6):** Apple_Fresh, Apple_Rotten, Banana_Fresh, Banana_Rotten, Orange_Fresh, Orange_Rotten.
**Input Size:** 224 × 224 pixels.

| Data Split | Percentage | Purpose |
| :--- | :--- | :--- |
| **Train** | ~80% | Updating model weights via backpropagation |
| **Validation** | ~10% | Evaluating performance during training to prevent overfitting |
| **Test** | ~10% | Final, unbiased evaluation of the best model |

**Data Leakage Prevention (Crucial Contribution):**
Many academic projects blindly shuffle data (Random Split). In agriculture datasets, burst photos create near-identical duplicates. If one duplicate lands in Train and another in Test, the model "memorizes" rather than learns (Data Leakage). 
*Our Solution:* We used **SHA-256** (for exact duplicates) and **dHash + Union-Find** (for similar images) to group identical fruits into physical groups. Entire groups were pushed to either Train, Val, or Test, ensuring the test set contains completely unseen physical fruits.

---

## 3. PREPROCESSING

To feed images into our PyTorch models, we applied specific transformations.

**Training Transforms (Includes Data Augmentation):**
*   **Resize(224, 224):** Standardizes input size for the models.
*   **RandomHorizontalFlip(p=0.5):** Flips the image randomly to increase dataset variety.
*   **RandomRotation(15 degrees):** Rotates the image slightly.
*   **ColorJitter(brightness, contrast, saturation=0.1):** Simulates different lighting conditions.
*   **ToTensor & Normalize:** Uses ImageNet statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

**Validation/Test Transforms:**
*   **Resize(224, 224), ToTensor, Normalize.**
*   *Note:* No data augmentation is applied to test images because we need an objective, undistorted evaluation of the model.

---

## 4. THE THREE MODELS

### 1. EfficientNetV2-S
*   **What is it?** A highly optimized Convolutional Neural Network (CNN) by Google.
*   **Why selected?** It balances accuracy and efficiency using Fused-MBConv blocks, making it highly accurate but faster to train.
*   **Inference Speed:** ~46.5 ms per image.
*   **Advantage:** Highest accuracy (98%) in our tests.

### 2. ConvNeXt-Tiny
*   **What is it?** A modernized, pure CNN by Meta that borrows architectural ideas from Transformers (like Swin).
*   **Why selected?** To test if a modernized CNN can beat standard CNNs. It uses massive 7x7 convolutional kernels (depthwise).
*   **Inference Speed:** ~50.3 ms per image.
*   **Advantage:** Excellent feature extraction, highly competitive with Transformers.

### 3. MobileViT-S
*   **What is it?** A hybrid lightweight model combining CNNs (MobileNetV2 blocks) and Vision Transformers (ViT) by Apple.
*   **Why selected?** To test Attention mechanisms (Transformers) on edge-device constraints. 
*   **Inference Speed:** ~36.7 ms per image (Fastest).
*   **Advantage:** Lowest parameter count (~5M), lightning fast.

---

## 5. MODEL ARCHITECTURE PIPELINE

Our unified model pipeline works as follows:

1.  **Pretrained Backbone:** (e.g., EfficientNet without its original top layer). It extracts a high-dimensional feature vector (e.g., 1280 numbers) from the 224x224 image.
2.  **LayerNorm:** Normalizes the feature vector for training stability.
3.  **Linear (Dense) Layer:** Projects features to an intermediate dimension.
4.  **GELU Activation:** Introduces non-linearity.
5.  **Dropout (0.3):** Randomly zeroes 30% of neurons during training to prevent overfitting.
6.  **Final Linear Layer (Logits):** Outputs exactly 6 numbers (one for each class).
7.  **Softmax:** Converts the 6 logits into probabilities that sum to 100%.

---

## 6. MODEL-SPECIFIC ARCHITECTURE SUMMARIES

*   **EfficientNetV2-S:** Uses Neural Architecture Search to find the best scaling of depth, width, and resolution. Uses Fused-MBConv to replace depthwise convolutions in early layers for speed.
*   **ConvNeXt-Tiny:** A pure CNN optimized to imitate Vision Transformers. Uses a patchified stem, large 7x7 depthwise convolutions, LayerNorm instead of BatchNorm, and GELU instead of ReLU.
*   **MobileViT-S:** Replaces local spatial processing with global processing using Transformers. It learns local features using CNNs, then unfolds them, applies Transformer attention for global context, and folds them back.

---

## 7. TRAINING

*   **Loss Function:** `CrossEntropyLoss` (Standard for multi-class classification).
*   **Optimizer:** `AdamW` (Adam with Weight Decay to prevent overfitting). Weight decay = 1e-4.
*   **Batch Size:** 32.
*   **Strategy:** Two-Stage Fine-Tuning (2 + 2 Epochs):
    *   **Stage 1 (2 Epochs):** Backbone is FROZEN. Only our custom Classification Head learns. LR = `1e-3`.
    *   **Stage 2 (2 Epochs):** Backbone is UNFROZEN. End-to-end fine-tuning. LR = `1e-4` (Smaller LR to avoid destroying pretrained weights).

---

## 8. EVALUATION & METRICS

*   **Accuracy:** Total correct predictions / Total predictions.
*   **Precision:** Out of all images the model *predicted* as "Rotten Apple", how many were actually Rotten Apples? (Reduces False Positives).
*   **Recall:** Out of all actual "Rotten Apples" in the dataset, how many did the model find? (Reduces False Negatives).
*   **F1-Score:** The harmonic mean of Precision and Recall.
*   **Macro F1:** The unweighted average of F1-scores across all 6 classes. *Why it matters:* It treats small classes equally to large classes, preventing class imbalance from hiding poor performance on minority classes.

---

## 9. CONFUSION MATRIX

*   **How to read it:** The Y-axis is the "True Class" (reality), and the X-axis is the "Predicted Class" (what the model guessed). The diagonal line represents correct predictions.
*   **Our Observations:** The models are highly accurate. Misclassifications rarely happen between fruit types (an Apple is rarely called a Banana). When errors occur, they are usually classifying a borderline "Fresh" fruit as "Rotten".

---

## 10. CONFIDENCE / UNCERTAIN PREDICTION

**Mechanism:** `CONFIDENCE_THRESHOLD = 0.60`
*   **Explanation:** The model outputs 6 probabilities. We take the `max()` probability. If this max probability is `< 0.60` (60%), the model is "Uncertain."
*   **Why use it?** If a user uploads an image of a dog, the model is forced to guess one of the 6 fruit classes. Because it has never seen a dog, its probabilities will be spread out (e.g., 30% Apple, 25% Banana). By checking if the highest score is below 60%, the web app warns the user: *"Low Confidence Prediction. A clearer image may produce a more reliable prediction."*

---

## 11. WEB APPLICATION

**Frontend (React + Vite):**
*   Provides a professional, responsive user interface.
*   *Pages:* Home, Scan (Upload & Camera), Dashboard (Metrics visualizations via Recharts), History (LocalStorage).
*   Sends images via standard HTTP `POST` requests using `FormData`.

**Backend (Flask API):**
*   Loads PyTorch models using a **Singleton ModelManager** (loads `.pth` files only once into memory at startup to prevent freezing).
*   *Endpoints:* 
    *   `POST /api/predict`: Receives image, processes tensor, returns prediction.
    *   `GET /api/metrics`: Returns JSON performance stats for the frontend dashboard.

---

## 12. COMPLETE PREDICTION FLOW (12 Steps)

1. User uploads a fruit image on the React frontend.
2. Frontend sends the image via `multipart/form-data` to Flask `/api/predict`.
3. Backend reads the byte stream into a PIL Image.
4. Preprocessing applies `Resize(224)` and `Normalize`.
5. Image is converted to a PyTorch Tensor: `[1, 3, 224, 224]`.
6. Tensor is passed through the selected model (e.g., MobileViT).
7. Backbone extracts a mathematical feature vector.
8. Classification head shrinks this vector to 6 logits.
9. Softmax converts logits into 6 percentage probabilities.
10. The highest probability is checked against the 0.60 threshold.
11. Backend returns a JSON response: `{fruit: "Apple", freshness: "Rotten", confidence: 98}`.
12. Frontend displays the result and saves it to local history.

---

## 13. PROJECT RESULTS

| Model | Parameters | Accuracy | Precision | Recall | Macro F1 | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNetV2-S** | 20.5M | **98.08%** | **99.28%** | **98.33%** | **98.75%** | 46.5 ms |
| **ConvNeXt-Tiny** | 28.0M | 94.23% | 95.16% | 96.82% | 95.89% | 50.3 ms |
| **MobileViT** | **5.1M** | 94.23% | 95.04% | 96.82% | 95.58% | **36.7 ms** |

**The Trade-off Explanation:**
EfficientNetV2 is the clear winner for **Accuracy**, achieving 98%. However, **MobileViT** is incredible because it achieves 94% accuracy while being **4x smaller** (5M vs 20M parameters) and significantly faster (36ms). MobileViT is the best choice if deployed on a mobile phone, whereas EfficientNet is best for cloud servers.

---

## 14. WHY THESE THREE MODELS?

*"We chose these three because they represent the three dominant paradigms in modern computer vision: EfficientNet represents highly optimized traditional CNNs, ConvNeXt represents pure CNNs modernized to compete with Transformers, and MobileViT represents the hybrid CNN-Transformer approach designed for lightweight edge devices."*

---

## 15. PROJECT NOVELTY / CONTRIBUTION

1.  **Data Leakage Prevention:** Implemented physical grouping using perceptual hashing (dHash) to ensure identical fruits don't cross train/test splits.
2.  **Architecture Comparison:** Benchmarked CNNs against Transformers on agricultural data.
3.  **Confidence-Aware Abstention:** Handled Out-of-Distribution (OOD) inputs dynamically.
4.  **End-to-End System:** Didn't just train a Jupyter Notebook model, but deployed a scalable Flask/React decoupled web architecture.

---

## 16. LIMITATIONS

1.  **Rectangular Stretching:** The model expects 224x224 squares. Uploading 16:9 phone photos stretches the fruit, slightly lowering real-world accuracy.
2.  **Limited Classes:** Only supports Apples, Bananas, and Oranges.
3.  **Background Noise Vulnerability:** Trained largely on clean backgrounds; hands or complex backgrounds can confuse the model.
4.  **No True OOD Detection:** We use a simple 0.60 confidence threshold instead of complex anomaly detection algorithms.
5.  **No Bounding Boxes:** It is an image classifier, not an object detector (YOLO); it cannot identify multiple different fruits in one image.

---

## 17. FUTURE WORK

1.  **Object Detection (YOLOv8):** Upgrade from classification to detection to locate rotten spots inside bounding boxes.
2.  **Grad-CAM Visualizations:** Generate heatmaps on the frontend to show *why* the model thinks a fruit is rotten.
3.  **Dataset Expansion:** Include Tomatoes, Mangos, Potatoes, and real-world supermarket backgrounds.
4.  **Aspect-Ratio Preserving Crop:** Fix the image stretching bug by using CenterCrop instead of direct Resize.
5.  **Edge Deployment:** Convert the MobileViT model to ONNX or TFLite and run inference directly in the browser via WebAssembly.

---

## 18. MOST IMPORTANT VIVA QUESTIONS

### Basic
**1. What is your project?** An AI web app that detects fruit type and freshness from images.
**2. What problem are you solving?** Automating the visual inspection of food quality to reduce waste.
**3. What is the input and output?** Input: RGB Image. Output: Fruit type, freshness (Fresh/Rotten), and confidence %.

### Dataset
**4. Which dataset did you use?** A customized version of the Top Indian Fruits Dataset.
**5. How many classes?** 6 (Apple/Banana/Orange, Fresh/Rotten).
**6. What is data leakage?** When near-identical images exist in both the Training and Testing sets, causing the model to memorize the test set instead of learning to generalize.
**7. How did you prevent leakage?** By using perceptual hashing (dHash) to group identical fruits and ensuring entire groups were kept in the same split.

### Deep Learning
**8. What is transfer learning?** Taking a model pre-trained on millions of images (ImageNet) and fine-tuning it on our specific fruit dataset.
**9. What is a backbone?** The main part of the neural network that extracts visual features (edges, textures) from the image.
**10. What is a classification head?** The final few layers we added to the backbone to convert the extracted features into our specific 6 classes.

### Models
**11. Why EfficientNetV2?** It uses Neural Architecture Search and Fused-MBConv to be extremely accurate yet computationally efficient.
**12. Why ConvNeXt?** To see if a modernized pure CNN can compete with Vision Transformers.
**13. Why MobileViT?** It combines local feature extraction of CNNs with the global attention mechanism of Transformers, making it lightweight.
**14. What is the trade-off?** EfficientNet is the most accurate (98%), but MobileViT is 4x smaller and faster, making it better for phones.

### Training
**15. What loss function did you use?** CrossEntropyLoss, standard for multi-class classification.
**16. Which optimizer?** AdamW, because it combines Adam's adaptive learning rate with strict weight decay to prevent overfitting.
**17. Why did you freeze the backbone initially?** To train our random classification head without destroying the valuable pre-trained weights in the backbone.

### Evaluation
**18. What is Macro F1?** The unweighted average of F1 scores across all classes. It prevents a majority class from inflating the score.
**19. Explain the confusion matrix.** It shows True classes vs. Predicted classes, allowing us to see exactly which fruits confuse the model.

### Application
**20. How does the frontend communicate with the backend?** Through REST APIs via HTTP POST requests sending JSON and Multipart Form Data.
**21. What happens when the model is uncertain?** If the max probability is below 60%, the app warns the user that the prediction has low confidence.

---

## 19. "EXPLAIN MY PROJECT IN 2 MINUTES"

"Good morning/afternoon. My project is **FreshGuard AI**, an automated food freshness detection system. In agricultural supply chains, manually sorting fresh and rotten produce is slow and labor-intensive. To solve this, we built a deep learning web application that instantly identifies if a fruit is fresh or rotten. 

We used a dataset of Apples, Bananas, and Oranges. A major challenge in agricultural datasets is **data leakage**—burst photos of the exact same fruit skew test results. We solved this by using perceptual hashing to ensure the model was tested on completely unseen physical fruits.

Instead of just building one model, we compared three modern architectures: **EfficientNetV2** (an optimized CNN), **ConvNeXt** (a modernized CNN), and **MobileViT** (a hybrid CNN-Transformer). We used Transfer Learning, fine-tuning them with a custom classification head using CrossEntropyLoss and AdamW.

Our results were excellent. EfficientNet achieved **98% accuracy**, while MobileViT achieved 94% but was 4 times smaller, proving it is ideal for mobile deployment. We then integrated these models into a Flask backend and built a React frontend. Users simply upload a photo, the backend processes the PyTorch tensor, calculates the Softmax probabilities, checks if the confidence is above 60% to reject random images, and returns the freshness result."

---

## 20. "IF THE FACULTY ASKS DEEPER"

**Q: Explain how MobileViT combines CNNs and Transformers.**
"CNNs are great at extracting local features (like edges and colors) but struggle to see the 'big picture' without many deep layers. Transformers use Self-Attention to look at the entire image globally at once, but they are very heavy. MobileViT uses standard MobileNet CNN blocks to extract local features, then 'unfolds' the image into patches, applies lightweight Transformer attention so the patches can talk to each other globally, and 'folds' it back. This gives us global context with CNN-like efficiency."

**Q: Explain your Two-Stage Fine-Tuning.**
"When you use a pretrained model, the backbone has learned excellent edge detection from ImageNet, but the classification head is randomly initialized. If we train both at once, the large errors from the random head will backpropagate and destroy the backbone's weights. So, in Stage 1, we freeze the backbone and only train the head with a high learning rate (1e-3). In Stage 2, we unfreeze everything and train end-to-end with a tiny learning rate (1e-4) to gently align the whole network to our fruits."

**Q: Why not use a standard Train/Test random split?**
"In burst photography, you might have 10 identical photos of the exact same rotten apple. If you do a random 80/20 split, 8 photos go to train, and 2 go to test. The model doesn't actually learn what 'rotten' looks like; it just memorizes that specific apple. By grouping based on hash similarity, all 10 photos stay together in the Train set, forcing the Test set to contain completely unique apples. This proves our 98% accuracy is real, not memorized."

---

## 21. QUICK REVISION CHEAT SHEET

| Category | Details |
| :--- | :--- |
| **Objective** | Detect Fruit Type and Freshness (Fresh/Rotten) via Deep Learning |
| **Dataset** | Top Indian Fruits (Subset) |
| **Classes** | Apple_Fresh, Apple_Rotten, Banana_Fresh, Banana_Rotten, Orange_Fresh, Orange_Rotten |
| **Input Size** | 224 x 224 (RGB Tensor) |
| **Loss & Optimizer** | CrossEntropyLoss / AdamW (Weight Decay 1e-4) |
| **Strategy** | Transfer Learning (2 Epochs Frozen + 2 Epochs Unfrozen) |
| **EfficientNetV2** | 20M params | 98.08% Acc | Best Overall Accuracy |
| **ConvNeXt-Tiny** | 28M params | 94.23% Acc | Modernized CNN baseline |
| **MobileViT** | 5M params | 94.23% Acc | Lightweight, Edge-ready |
| **Backend** | Flask API (PyTorch inference, Singleton Model loading) |
| **Frontend** | React + Vite (Web UI, Confidence Warning) |
| **Confidence Threshold** | `< 0.60` max Softmax probability = "Uncertain" warning |
| **Main Novelty** | Leakage-safe dataset splitting via Perceptual Hashing (dHash) |
| **Main Limitation** | 224x224 Resize physically distorts 16:9 phone photos |
| **Future Work** | Upgrade to Object Detection (YOLO) and add Grad-CAM heatmaps |
