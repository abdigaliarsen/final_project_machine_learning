Project Delivery:

I'd like to use Docker here, but let's do run the project with bash only scripts first:


source env/bin/activate
pip install -r requirements.txt

login to your kaggle profile, get API key from your account settings (create it if you don't have one)
then download kaggle.json credentials and move then to ~/.kaggle/ folder

join the competition: https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview

after that proceed with the next commands:

kaggle competitions download -c sentiment-analysis-on-movie-reviews

