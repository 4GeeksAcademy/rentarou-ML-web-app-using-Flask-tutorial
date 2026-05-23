from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from pathlib import Path

app = Flask(__name__)


BASE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BASE_DIR.parent


MODEL_PATH = PROJECT_ROOT / "data" / "models" / "anime_high_score_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "data" / "models" / "model_features.pkl"


model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)



@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    probability_text = None

    if request.method == "POST":
        try:
           

            num_list_users = int(request.form["num_list_users"])
            num_scoring_users = int(request.form["num_scoring_users"])
            num_episodes = int(request.form["num_episodes"])
            average_episode_duration = int(request.form["average_episode_duration"])

            media_type = request.form["media_type"]
            status = request.form["status"]
            rating = request.form["rating"]

            

            input_data = pd.DataFrame([{
                "num_list_users": num_list_users,
                "num_scoring_users": num_scoring_users,
                "num_episodes": num_episodes,
                "average_episode_duration": average_episode_duration,
                "media_type": media_type,
                "status": status,
                "rating": rating
            }])

            
            input_data = input_data[features]

            

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

           

            if prediction == 1:
                prediction_text = "The model predicts this anime is likely to have a high score."
            else:
                prediction_text = "The model predicts this anime is not likely to have a high score."

            probability_text = f"Probability of high score: {probability:.2%}"

        except Exception as e:
            prediction_text = f"Error processing the prediction: {str(e)}"
            probability_text = None

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        probability_text=probability_text
    )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )