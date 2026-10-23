# Smart Crop Recommendation — Upgrade Pack

## Files
- `app.py`: upgraded Streamlit app
- `requirements.txt`: Python dependencies

## Install
Run these commands from your existing project folder (the same folder containing `app.py` and `crop_model.pkl`):

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep your existing `crop_model.pkl` beside `app.py`. Do not delete your current working project; make a backup before replacing `app.py`.

## Dataset for evaluation
Place your labeled crop CSV beside `app.py`. The app looks for:
- `Crop_recommendation.csv`
- `crop_recommendation.csv`
- `dataset.csv`

Expected columns: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, and `label`.

## Features
- Dark agriculture dashboard
- Crop recommendation with model output scores if `predict_proba` is supported
- Model-native feature importance when available
- Local SQLite prediction history and CSV export
- PDF and CSV recommendation reports
- Optional Open-Meteo place/weather lookup
- Evaluation metrics and confusion matrix when the labeled dataset is present

## Important limitations
- Model output scores are not necessarily calibrated probabilities.
- Current precipitation is not seasonal rainfall; weather values must be reviewed.
- Feature importance is not a causal explanation.
- Evaluation against a CSV that was also used to train the model is not an independent test. Use a proper held-out test set for reported performance.
- The app does not guarantee 100% correct predictions.
