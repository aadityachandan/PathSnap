"""Laptop 1. Prediction reads .track/.nodes/.arcs ONLY; scoring runs afterwards."""
import os,re,json,time,logging
from pathlib import Path
from threading import Lock
import joblib,numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from pathsnap.data import load_trip,candidates
from pathsnap.context import model_input
from pathsnap.decoder import decode,geometry
from pathsnap.maps import make_map

PROJECT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('PATHSNAP_TRIPS',PROJECT/'data'/'student_test')).resolve()
REFERENCES=os.environ.get('PATHSNAP_REFERENCES')
ARTIFACTS=Path(os.environ.get('PATHSNAP_ARTIFACTS',PROJECT/'artifacts'))
app=FastAPI(title='PathSnap API',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
log=logging.getLogger('uvicorn.error');lock=Lock();model=None;metadata=None

class MatchRequest(BaseModel):
    trip_id:str=Field(pattern=r'^\d{8}$')


def available():
    if not DATA.is_dir():return []
    return sorted(d.name for d in DATA.iterdir() if d.is_dir() and re.fullmatch(r'\d{8}',d.name)
        and all((d/f'{d.name}.{e}').is_file() for e in ['track','nodes','arcs']))

@app.get('/health')
def health():
    return {'status':'ok','model_ready':all((ARTIFACTS/name).is_file() for name in ('final_model.joblib','metadata.json')),'trip_count':len(available())}

@app.get('/trips')
def trips():return {'trip_ids':available(),'mode':'evaluation' if DATA.name!='student_test' else 'student development'}

@app.post('/match')
def match(request:MatchRequest):
    global model,metadata
    if request.trip_id not in available():raise HTTPException(404,'Unknown trip')
    # Bound simultaneous CPU/memory use on a laptop.
    if not lock.acquire(blocking=False):raise HTTPException(409,'Another trip is processing. Retry after it finishes.')
    try:
        if model is None:
            if not all((ARTIFACTS/name).is_file() for name in ('final_model.joblib','metadata.json')):
                raise HTTPException(503,'Install both final_model.joblib and metadata.json first')
            # Publish the cache only after both files load successfully.
            loaded_metadata=json.loads((ARTIFACTS/'metadata.json').read_text())
            loaded_model=joblib.load(ARTIFACTS/'final_model.joblib')
            model,metadata=loaded_model,loaded_metadata
        start=time.perf_counter();tr=load_trip(DATA/request.trip_id);ids,features=candidates(tr)
        p=model.predict_proba(model_input(tr,ids,features))[:,1].reshape(ids.shape)
        predicted,pieces,breaks=decode(tr,ids,features,p,**metadata['decoder'])
        prediction_seconds=time.perf_counter()-start
        response={'trip_id':tr.id,'observations':len(tr.track),'model':metadata['selected_model'],
            'prediction_seconds':prediction_seconds,'predicted_arc_ids':predicted.tolist(),
            'route_latlon':geometry(tr,pieces),'gps_latlon':tr.track[:,[1,0]].tolist(),
            'break_count':len(breaks),'break_observations':breaks,'metrics':None,
            'map_html':make_map(tr,pieces).get_root().render()}
        # Separate reference-only evaluation, AFTER inference, no labels in response.
        if REFERENCES:
            from pathsnap.reference import load_reference,align_reference
            from pathsnap.metrics import route_metrics
            refpath=Path(REFERENCES)/tr.id/f'{tr.id}.route'
            if refpath.is_file():
                route=load_reference(refpath.parent,len(tr.arcs));labels,_=align_reference(tr,route)
                response['metrics']=route_metrics(tr,predicted,labels,route,[a for piece in pieces for a in piece])
                response['metric_note']='Pointwise accuracy uses derived monotone alignment; not independent GPS-level labels.'
        log.info('MATCH trip=%s observations=%d prediction_seconds=%.3f breaks=%d',tr.id,len(tr.track),prediction_seconds,len(breaks))
        return response
    finally:lock.release()
