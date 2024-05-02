import pandas as pd
from fastapi import FastAPI
from src.recommendation.recommendation import GameRecommendation  
from src.recommendation.ann_recommendation import ApproximateNearestNeighbors

app = FastAPI()

game_db = pd.read_parquet("data/processed/game_database.parquet")
cos_sim = pd.read_parquet("data/metrics/cos_sim_features.parquet")

game_recommendation = GameRecommendation(game_db, cos_sim)
ann = ApproximateNearestNeighbors(game_db)

@app.get("/")
def index():
    return {"Project":"Game Content Based Recommender System"}

@app.get("/search")
def read_item(game_name : str):
    game = game_db[game_db['name'] == game_recommendation.game_finder(game_name)]
    return game.to_json(orient="records")

@app.get("/recommend")
def recommend(game_name : str):
    return game_recommendation.get_recommendation(game_name)

@app.get("/recommend_img")
def recommend_img(game_name : str):
    return game_recommendation.get_recommendation_img(game_name)

@app.get("/recommend_ann")
def recommend_ann(game_name : str):
    embeddings = game_db['embedding'].tolist()
    ann.build_index(embeddings)
    return ann.get_recommendations(game_name)