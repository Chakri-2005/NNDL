import os
from pptx import Presentation
from pptx.util import Inches, Pt

def add_slide(prs, title, content="", layout_idx=1):
    slide_layout = prs.slide_layouts[layout_idx]
    slide = prs.slides.add_slide(slide_layout)
    title_placeholder = slide.shapes.title
    title_placeholder.text = title
    if content and layout_idx == 1:
        body_placeholder = slide.placeholders[1]
        body_placeholder.text = content
    return slide

def generate_pptx():
    prs = Presentation()
    
    # 1. Title
    slide = add_slide(prs, "FreshGuard AI", layout_idx=0)
    slide.placeholders[1].text = "Review 3 Rubric Action Plan\nTeam Verdura"

    # 2. Team + What's New
    add_slide(prs, "Team Verdura & What's New", 
              "Team:\n- Kousik Sarma Lakkaraju\n- Manikanta Sai\n- Chakravarthy Vemula\n\nWhat's New:\n- Dataset updated to IEEE Dataport FruitsGB\n- Flagship Architecture: Attention-CNN\n- Full literature survey with Scimago verified papers")

    # 3. Problem Statement & Motivation
    add_slide(prs, "Problem Statement & Motivation", 
              "Problem: Manual inspection of fruit quality is slow, subjective, and inconsistent.\n\nMotivation: Food safety and waste reduction require on-device, accessible tracking without cloud dependency.\n\nObjective: Real-time, offline, safety-first freshness classifier.")

    # 4. Literature Survey
    add_slide(prs, "Literature Survey", 
              "We surveyed 15 papers across Q1/Q2 Scimago journals (e.g., CVPR, TPAMI, IEEE Access).\n\nKey areas:\n1. Attention Mechanisms in CNNs (e.g., CBAM, SE-Net)\n2. Hyperparameter Tuning (Optuna, BayesOpt) for Imbalanced Data\n3. Edge Deployment (Quantization, MobileNet) & Sequence Models (GRU)\n\n(See full survey table in report for exact quartiles and findings)")

    # 5. Overall Architecture Diagram
    add_slide(prs, "Overall Architecture Diagram", 
              "[Insert Overall Lane Diagram Here]\n\nDevice Input -> Pre-processing -> Model -> Safety Layer -> Output")

    # 6. Module Details
    add_slide(prs, "Module Details", 
              "- Capture Module: Device Camera API (RGB Frame)\n- Pre-processing: Normalization (Cleaned Tensor)\n- Classification Module: Attention-CNN INT8 (Probabilities)\n- Safety Module: OOD threshold logic (Verdict/Abstain)\n- UI Module: App Front-end (Result Screen)")

    # 7. Dataset Description
    add_slide(prs, "Dataset Description (FruitsGB)", 
              "Source: IEEE Dataport (DOI: 10.21227/gzkn-f379)\n\nScope Decision:\nFocused on Apple, Banana, and Orange (Fresh/Rotten) to ensure our Attention-CNN learns highly robust freshness features rather than generic cosmetic defects. This explicitly aligns with our food safety problem statement.\n\nSplit: 6000 images total (70% Train, 15% Val, 15% Test)")

    # 8. Deep Learning Architecture
    add_slide(prs, "Deep Learning Architecture", 
              "Attention-CNN with Safety Fallback\n\nNovelty: Combines a lightweight 3-block CNN with a CBAM-style attention gate tuned for spoilage, plus a joint confidence-abstention layer for safety-critical deployment.\n\nComplexity (from torchinfo):\n- Total Parameters: ~104k\n- Estimated FLOPs: 1.33 GFLOPs\n- Estimated Size: 0.42 MB (FP32)")

    # 9. Algorithm Procedure
    add_slide(prs, "Algorithm Procedure", 
              "1. Forward Pass:\n   - Convolution & ReLU\n   - CBAM Attention (Spatial & Channel gating)\n   - Global Average Pooling -> Dense\n\n2. Training:\n   - Loss: CrossEntropy\n   - Optimizer: Adam (lr=1e-4)\n\n3. Safety Layer:\n   - If max(Softmax(logits)) < threshold -> Abstain")

    # 10. Hyperparameter Table
    add_slide(prs, "Hyperparameter Details", 
              "- Optimizer: Adam (Proven fast convergence for image tasks)\n- Learning Rate: 1e-4 (Standard for lightweight CNNs)\n- Batch Size: 32 (Balances memory and gradient stability)\n- Dropout: 0.3 (Prevents overfitting on 6k images)\n- Confidence Threshold: 0.6 (Safety fallback threshold)")

    # 11. Performance Metrics
    add_slide(prs, "Performance Metrics", 
              "Evaluated on unseen Test Set (15%):\n\n- Accuracy: Overall correctness\n- Precision: Minimizes false spoilage flags\n- Recall: Minimizes missed rotten fruit (critical for safety)\n- Macro-F1: Balances class imbalances\n\n(Literature benchmark target: ~98% from EfficientNet OOD systems)")

    # 12. Results & Discussion
    res_slide = add_slide(prs, "Results & Discussion", "Plots generated from actual training run.")
    # We will try to add images if they exist
    if os.path.exists("results/training_curves.png"):
        res_slide.shapes.add_picture("results/training_curves.png", Inches(1), Inches(2), width=Inches(4))
    if os.path.exists("results/confusion_matrix.png"):
        res_slide.shapes.add_picture("results/confusion_matrix.png", Inches(5.5), Inches(2), width=Inches(4))

    # 13. UI Screens
    ui_slide = add_slide(prs, "UI Screens (Mockups)", "")
    
    mockup_dir = "slides/ui_mockups/screenshots"
    if os.path.exists(mockup_dir):
        # We assume 4 screenshots are present
        capture = os.path.join(mockup_dir, "capture_screen.png")
        result = os.path.join(mockup_dir, "result_screen.png")
        abstain = os.path.join(mockup_dir, "abstain_screen.png")
        tracker = os.path.join(mockup_dir, "tracker_screen.png")
        
        # Position them side by side
        # Slide width is 10 inches, so each gets ~2.2 inches width with some spacing
        width = Inches(2.2)
        top = Inches(1.5)
        
        if os.path.exists(capture):
            ui_slide.shapes.add_picture(capture, Inches(0.2), top, width=width)
        if os.path.exists(result):
            ui_slide.shapes.add_picture(result, Inches(2.6), top, width=width)
        if os.path.exists(abstain):
            ui_slide.shapes.add_picture(abstain, Inches(5.0), top, width=width)
        if os.path.exists(tracker):
            ui_slide.shapes.add_picture(tracker, Inches(7.4), top, width=width)
    else:
        ui_slide.placeholders[1].text = "Mockups designed in HTML/CSS:\n1. Capture Screen (Camera Viewfinder)\n2. Result Screen (Fresh/Rotten + Confidence)\n3. Abstain Screen (Safety Fallback for low confidence)\n4. Tracker Screen (GRU timeline for batches)\n\n(See /slides/ui_mockups for live HTML files)"

    # 14. Standard Paper Chosen
    add_slide(prs, "Standard Paper Chosen", 
              "Title: Deep Learning based Food Quality Inspection using Attention Networks (2022)\nJournal: Computers and Electronics in Agriculture (Q1)\n\nWhy: Closest match utilizing lightweight attention CNNs for fruit grading.\nNovelty vs Baseline: We target consumer edge devices with an explicitly designed OOD abstention safety layer, whereas the baseline targets industrial sorting belts.")

    # 15. Summary & Next Steps
    add_slide(prs, "Summary & Next Steps", 
              "Summary:\n- Dataset secured and processed\n- Model trained with strong baseline results\n- UI mockups ready\n\nNext Steps:\n- Optimize INT8 Quantization on device\n- Integrate GRU tracker logic for v2")

    prs.save('slides/FreshGuard_Review3.pptx')
    print("Successfully generated slides/FreshGuard_Review3.pptx")

if __name__ == "__main__":
    generate_pptx()
