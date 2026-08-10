from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routers import (
    area_yield,
    cat_bond,
    credibility,
    extreme_value,
    health_micro,
    motor,
    portfolio,
    premium,
    reinsurance,
    reserving,
    risk_metrics,
    ruin,
    runs,
    sensitivity,
    simulation,
    weather_index,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Actuarial Risk Model API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


for router in (
    premium.router,
    simulation.router,
    risk_metrics.router,
    reinsurance.router,
    reserving.router,
    credibility.router,
    portfolio.router,
    extreme_value.router,
    ruin.router,
    sensitivity.router,
    runs.router,
    weather_index.router,
    area_yield.router,
    cat_bond.router,
    motor.router,
    health_micro.router,
):
    app.include_router(router)
