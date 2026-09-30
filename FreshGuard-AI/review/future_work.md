# FreshGuard AI — Limitations & Future Roadmap

## 1. Project Limitations
1. **Scope Restriction**: Currently tuned and validated on 6 key fruit/freshness categories (`Apple_Fresh`, `Apple_Rotten`, `Banana_Fresh`, `Banana_Rotten`, `Orange_Fresh`, `Orange_Rotten`). Expanding to exotic or multi-fruit produce requires new group-isolated data collection.
2. **RGB Modality Constraints**: Visual RGB inspection detects surface rot, bruising, and skin discoloration, but cannot detect internal flesh decay without multispectral imaging sensors.
3. **Lighting Variation Sensitivity**: While data augmentation mitigates minor illumination shifts, extreme glare or heavy shadows under uncalibrated smartphone flash can decrease prediction confidence.

## 2. Future Development Roadmap

### Phase 1: Post-Training Quantization (PTQ) & Edge Acceleration
* Convert trained PyTorch models (`.pth`) to **ONNX Runtime** and **TensorLite (TFLite) INT8 Quantized** formats.
* Target hardware acceleration via Android Neural Networks API (NNAPI) and iOS CoreML to achieve $< 15\text{ ms}$ inference latency on modern smartphones.

### Phase 2: Explainability & Grad-CAM Visual Inspection
* Integrate Gradient-weighted Class Activation Mapping (Grad-CAM) into the Flask API backend.
* Render interactive heatmaps over uploaded fruit photos, highlighting exact visual regions (e.g. skin rot spots or soft bruises) that triggered the spoilage verdict.

### Phase 3: Hyperspectral & Time-Series Quality Tracking
* Extend the vision pipeline to process multi-spectral camera feeds.
* Integrate recurrent sequence tracking to model fruit shelf-life decay trajectories over multi-day kitchen storage cycles.
