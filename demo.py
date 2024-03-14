from fastapi import FastAPI, Cookie

app = FastAPI()

@app.get("/set-cookie/")
async def set_cookie(param: str):
    response = {"message": "Cookie set successfully"}
    return response, {"param": param}

@app.get("/get-cookie/")
async def get_cookie(param: str = Cookie(None)):
    return {"param": param}
