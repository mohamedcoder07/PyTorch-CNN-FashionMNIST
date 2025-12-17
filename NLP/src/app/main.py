import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel

import joblib



app = FastAPI(title= "Amazon Sentiment Analysis API")


class Review(BaseModel):
    text: str
    sentiment: str

all_reviews = []

model_path = "saved_models/model_lightgbm.pkl"
vectorizer_path = "saved_models/tfidf_vectorizer.pkl"

classifier = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)



@app.get("/")
def data_description():
    return ""



@app.post("/predict-sentiment", response_model=str)
async def predict_sentiment(review: str):
    embed = vectorizer.transform([review])
    prediction = classifier.predict(embed)[0]
    # item = Review(text = review, sentiment = prediction)
    # all_reviews.append(item)
    # {"Review" : item.text,
    #         "Sentiment" : item.sentiment}
    return prediction



# @app.post("/predict-sentiment", response_model=[str, File])
# async def predict_sentiment(review: str|UploadFile):
#     if type(review) == str:
#         embed = vectorizer.transform([review])
#         prediction = classifier.predict(embed)[0]
#         all_reviews.append(Review(text = review, sentiment = prediction))
#         # {"Review" : item.text,
#         #         "Sentiment" : item.sentiment}
#         return prediction
#     elif type(review) == UploadFile:




# @app.post("/predict-sentiment")
# async def predict_sentiment(file: UploadFile = File(...)):
#     content = await file.read()

#     return ""   






if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)