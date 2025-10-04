# 🚀 EXOplorer — The Exoplanet Intelligence Explorer

> “We didn’t just teach an AI to discover new worlds — we taught it to explain how it found them.”
## Pipeline
![Pipeline Architecture](Pipeline%20Architecture.png)

---

## 🌍 Project Summary

**EXOplorer** is an **AI-powered exoplanet classification and exploration platform** built to connect **scientists** and the **public** through transparent and interpretable AI.  

It classifies exoplanet candidates as **Confirmed**, **Candidate**, or **False Positive**, using machine learning models trained on NASA’s **KOI**, **TOI**, and **K2** datasets.  
The system also explains *why* it made each prediction with **SHAP-based interpretability**, ensuring that results are both **accurate and scientifically explainable**.

Beyond classification, EXOplorer features an engaging **Hunt Planets Game**, bridging the gap between researchers and the public by teaching users about real NASA-discovered exoplanets through interactive challenges and a global leaderboard.

---

## 🎥 Project Demonstration

- 🎬 **Video Demo (30 seconds):** [Watch Here](#)  
- 🌐 **Live Website:** [Visit EXOplorer](#)  
- 💻 **GitHub Repository:** [Explore the Code](#)

---

## 🛰️ Project Details

### 🔑 Key Features

#### 🧮 AI Classification & Transparency Dashboard
A unified, interactive interface combining AI prediction and interpretability.  

- Upload candidate data (CSV or manual input) to the `/classify` API.  
- The backend returns **prediction**, **confidence**, and **SHAP feature-attribution values**.  
- The frontend visualizes this information in a **dynamic AI Transparency Dashboard** with animated bar charts and tooltips explaining the *top-5 most influential features*.  

This feature empowers scientists to audit AI reasoning — and helps non-experts intuitively understand how features like **orbital period**, **planet radius**, or **stellar temperature** affect classification.

---

#### 🔬 Scientist Dashboard

A powerful, insight-rich dashboard designed for researchers and domain experts.  
It aggregates and visualizes metrics from multiple NASA datasets to give scientists an instant overview of current discoveries and trends.

Through **interactive charts** and **advanced, filterable tables** for the KOI, TOI, and K2 datasets, scientists can explore planetary distributions, compare dataset dispositions, and isolate subsets of interest — such as small rocky planets or those in habitable zones.  

This dynamic dashboard transforms static data into **actionable scientific insights**, reinforcing **reproducibility, exploration, and discovery**.

**Dashboard Highlights:**
- Confirmed Planets: 6,007  
- Awaiting Candidates: 8,000+  
- Integrated Datasets: KOI, TOI, K2  
- Unified Data Explorer:  
  - Planet Radius Distribution  
  - Orbital Period Distribution  
  - Equilibrium Temperature  
  - Stellar Distance Distribution  
  - Dataset Comparison by Disposition  
  - Disposition Breakdown  
  - Discoveries Over Time  

---

#### 🎮 Hunt Planets Game — *Where Science Meets Curiosity*

A gamified, educational experience designed to **bridge the gap between professional scientists and the public**.  

- Players explore *real NASA-discovered exoplanets* from past missions.  
- The game asks engaging questions about each planet’s characteristics, orbit, or discovery mission.  
- Players gain points for correct answers and speed, climbing a **global leaderboard** that encourages competition and learning.  

This experience turns NASA’s open exoplanet data into an **interactive educational journey**, empowering the public to engage with real space science while learning how exoplanets are identified and studied.

---

#### ⚙️ Retrain-on-the-Fly (with Hyperparameter Tuning)

A **scientist-focused retraining interface** for experimentation and model iteration.  

- Upload labeled datasets and retrain the classification model directly through the `/api/train` endpoint.  
- Fine-tune hyperparameters (learning rate, estimators, max depth, etc.) to test new configurations.  
- Updated model artifacts and metrics are saved, reloaded, and visualized automatically.

This feature creates a **continuous improvement loop**, allowing scientists to refine model performance, validate results, and adapt to new datasets or missions.

---

### 💡 Benefits

- **Scientific Transparency:** SHAP-powered explanations make AI predictions interpretable and auditable.  
- **Reproducibility:** Fully documented ML lifecycle with retraining, versioning, and metadata tracking.  
- **Engagement & Education:** The “Hunt Planets” game brings NASA’s discoveries to life for the public.  
- **Flexibility:** Backend can integrate new NASA missions or additional datasets easily.  
- **Mission-Agnostic Generalization:** Designed to generalize across datasets, adaptable to future NASA or partner missions without major changes.  
- **Accessibility:** Intuitive UI for students and educators, yet powerful enough for professional researchers.

---

## 🧩 NASA Data Usage

**Datasets Used:**
- [Kepler Object of Interest (KOI) Catalog](https://exoplanetarchive.ipac.caltech.edu/)  
- [Transiting Exoplanet Survey Satellite (TOI) Catalog](https://exoplanetarchive.ipac.caltech.edu/docs/TESS.html)  
- [K2 Mission Archive](https://keplerscience.arc.nasa.gov/k2-data-release-notes.html)

**Usage:** These datasets provide planetary and stellar parameters such as **radius**, **orbital period**, **equilibrium temperature**, and **stellar magnitude**, forming the foundation for model training, testing, and visual exploration.

---

## 🛰️ Space Agency Partner & Other Data

**External Tools & Libraries:**
- [NASA Exoplanet Archive API](https://exoplanetarchive.ipac.caltech.edu/)  
- [scikit-learn](https://scikit-learn.org/) – ML pipeline management  
- [LightGBM](https://lightgbm.readthedocs.io/) & [XGBoost](https://xgboost.readthedocs.io/) – stacked ensemble classifiers  
- [SHAP](https://github.com/slundberg/shap) – explainable AI visualizations  
- [pandas](https://pandas.pydata.org/) & [numpy](https://numpy.org/) – data preprocessing  
- [React](https://react.dev/) + [Chart.js](https://www.chartjs.org/) – frontend visualizations  
- [Framer Motion](https://www.framer.com/motion/) – animations  
- [Supabase](https://supabase.com/) – leaderboard backend  

All resources used are **open-source or publicly available**, with **no copyrighted or restricted materials**.

---

## 🤖 Use of Artificial Intelligence (AI)

During development, the team used several **AI-assisted tools** to accelerate development and enhance productivity:

- 💬 **[ChatGPT (OpenAI)](https://chat.openai.com/)** — for code generation, documentation refinement, debugging, and design assistance.  
- 💡 **[GitHub Copilot](https://github.com/features/copilot)** — for intelligent code completion and boilerplate generation.  
- 🧠 *(Optional)* **Gemini (Google)** — for quick ideation and feature design brainstorming.

> These AI tools supported the **development process**, but all core logic, ML modeling, and scientific validation were performed and verified by the team.

---

## ⚙️ Tech Stack

### 🧠 Backend
- Python 3.x  
- Flask (API endpoints)  
- scikit-learn / LightGBM / XGBoost  
- SHAP for explainability  
- pandas, numpy, scipy  
- joblib (model persistence)

### 💻 Frontend
- React + TypeScript  
- Vite (bundling)  
- Chart.js + react-chartjs-2 (visualizations)  
- Framer Motion (animations)  
- lucide-react (icons)  
- PapaParse (CSV parsing)  
- Supabase (leaderboard data)

---

## 🧾 Setup & Installation

### 1️⃣ Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2️⃣ Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

- Access the frontend at http://localhost:5173/

- Access the backend API at http://localhost:5000/

--- 
# 🌠 Closing Note
EXOplorer unites AI, science, and education in a single mission: to explore, understand, and trust the discoveries that expand humanity’s view of the universe.
Built on NASA’s open data, powered by explainable AI, and designed to inspire the next generation of explorers.

- 👨‍🚀 Team: PathFinders
- 📅 Hackathon: NASA Space Apps Challenge 2025
- 🛰️ Challenge: A World Away: Hunting for Exoplanets with AI
