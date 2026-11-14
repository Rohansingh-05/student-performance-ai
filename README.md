# Student Performance AI — Vercel starter

This migration replaces Streamlit with a Next.js frontend and FastAPI model API. It uses your existing trained scikit-learn files; it does not retrain or invent models.

## Architecture
- `frontend/`: Next.js dashboard, deployed as a Vercel project.
- `backend/`: FastAPI API and model artifacts, deployed as a second Vercel project.

## Copy existing artifacts
Copy these from your current Streamlit project into `backend/models/`:
- `decision_tree.pkl`
- `random_forest.pkl`
- `imputer.pkl`
- `evaluation_results.json`

Copy `data/student_performance_dataset.csv` into `backend/data/` if you want dataset summary data.

**Pickle compatibility:** in your current Windows virtual environment, run:
```powershell
python -c "import sklearn,numpy,pandas; print('scikit-learn',sklearn.__version__); print('numpy',numpy.__version__); print('pandas',pandas.__version__)"
```
Pin those exact versions in `backend/requirements.txt` before deployment. Only load pickle files you created/trust.

## Run backend locally (PowerShell)
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Test `http://127.0.0.1:8000/health` and `/docs`.

## Run frontend locally (second PowerShell)
```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
```
Set `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```
Then run `npm run dev` and open `http://localhost:3000`.

## Deploy on Vercel
Deploy as **two Vercel projects from the same GitHub repository**:
1. Backend project: Root Directory `backend`. Vercel detects the FastAPI app in `main.py`. Deploy and test `/health` and `/docs`.
2. Frontend project: Root Directory `frontend`. Add `NEXT_PUBLIC_API_URL` with the backend deployment URL, e.g. `https://your-api.vercel.app`, then deploy.

Set backend environment variable `FRONTEND_ORIGIN` to your actual frontend URL before public production use. The starter allows all origins for initial testing.

## AI features
- Existing Decision Tree and Random Forest inference.
- Compare both predictions and indicate agreement/disagreement.
- Class probabilities.
- Existing test metrics and confusion matrices.
- Global feature importance from each fitted tree model.
- Batch prediction endpoint.
- Rule-based support prompts based on low-scoring indicators (clearly separated from model predictions).

Feature importance is global model reliance, not causal explanation of an individual result. Add SHAP only after first deployment and validate its outputs.

## Caveat
The current dataset and labels are synthetic. Test scores only describe the synthetic held-out split. This is a decision-support prototype, not a final student assessment system.
