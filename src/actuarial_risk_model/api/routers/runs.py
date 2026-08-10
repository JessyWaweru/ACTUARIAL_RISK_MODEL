import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import Run, get_db
from ..schemas import RunCreateRequest, RunDetail, RunSummary

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunSummary)
def create_run(req: RunCreateRequest, db: Session = Depends(get_db)) -> Run:
    run = Run(
        kind=req.kind,
        name=req.name,
        input_json=json.dumps(req.input),
        result_json=json.dumps(req.result),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.get("", response_model=list[RunSummary])
def list_runs(kind: str | None = None, db: Session = Depends(get_db)) -> list[Run]:
    stmt = select(Run).order_by(Run.created_at.desc())
    if kind:
        stmt = stmt.where(Run.kind == kind)
    return list(db.execute(stmt).scalars())


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.to_detail_dict()


@router.delete("/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)) -> dict:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete(run)
    db.commit()
    return {"deleted": run_id}
