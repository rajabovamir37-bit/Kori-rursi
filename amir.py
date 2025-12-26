from fastapi import FastAPI
app = FastAPI(
    title="Магазин техники",
    description="Backend веб-приложения для интернет-магазина техники",
    version="1.0.0"
)
@app.get("/")
def root():
    return {"message": "Backend магазина техники запущен"}
