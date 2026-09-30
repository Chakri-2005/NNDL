cd C:\NNDL

# 1
git add FreshGuard-AI/config.py FreshGuard-AI/requirements.txt FreshGuard-AI/.gitignore
git commit -m "feat: Project configuration and dependencies"

# 2
git add FreshGuard-AI/README.md FreshGuard-AI/literature/ FreshGuard-AI/slides/ FreshGuard-AI/Review3_Rubric_Plan.md FreshGuard-AI/*.xlsx
git commit -m "docs: Documentation, literature survey, and presentation slides"

# 3
git add FreshGuard-AI/dataset.py FreshGuard-AI/scripts/
git commit -m "feat: Dataset preprocessing and split generation scripts"

# 4
git add FreshGuard-AI/models/
git commit -m "feat: Deep learning model architectures (EfficientNet, ConvNeXt, MobileViT)"

# 5
git add FreshGuard-AI/review/
git commit -m "feat: Training logs and review evaluation tools"

# 6
git add FreshGuard-AI/results/metrics/ FreshGuard-AI/results/per_class/
git commit -m "chore: Final model metrics and evaluation results"

# 7
git add FreshGuard-AI/backend/
git commit -m "feat: Flask backend API for robust model serving"

# 8
git add FreshGuard-AI/frontend/package.json FreshGuard-AI/frontend/vite.config.js FreshGuard-AI/frontend/index.html FreshGuard-AI/frontend/src/main.jsx FreshGuard-AI/frontend/src/App.jsx FreshGuard-AI/frontend/src/styles/
git commit -m "feat: Frontend React+Vite architecture and global styling"

# 9
git add FreshGuard-AI/frontend/src/components/ FreshGuard-AI/frontend/src/services/
git commit -m "feat: Frontend API services and UI components"

# 10
git add FreshGuard-AI/frontend/src/pages/
git commit -m "feat: Frontend interactive pages (Scan, Dashboard, History, etc.)"

# 11
git add FreshGuard-AI/create_*.py FreshGuard-AI/fix_frontend.py FreshGuard-AI/app_old.py.bak
git commit -m "chore: Development automation and scaffolding scripts"

# 12
git add .
git commit -m "chore: Final cleanup and minor adjustments"

# Push all to remote
git push origin main
