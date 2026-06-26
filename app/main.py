from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import init_db
from app.embeddings import get_model
from app.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_model()  # load the model at startup so the first request isn't slow
    yield


app = FastAPI(title="Semantic Document Search", lifespan=lifespan)
app.include_router(router)
