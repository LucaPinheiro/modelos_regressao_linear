from pydantic import BaseModel
from fastapi import FastAPI
import joblib

app = FastAPI()

class Pontuacao_Request(BaseModel):
    horas_estudo: float

class Pontuacao_Response(BaseModel):
    pontuacao: int

modelo_pontuacao = joblib.load('./modelos_treinados/modelo_regressao_pontuacao.pkl')

@app.post('/predict', response_model=Pontuacao_Response)
def predict(data: Pontuacao_Request) -> Pontuacao_Response:
    input_feature = [[data.horas_estudo]]

    y_pred = modelo_pontuacao.predict(input_feature)[0]

    return Pontuacao_Response(pontuacao=int(y_pred))
