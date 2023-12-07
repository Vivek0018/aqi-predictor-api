# main.py

from fastapi import FastAPI



app = FastAPI()

@app.get("/", tags=['ROOT'])
async def root():
    return {"message": "Hello World"}