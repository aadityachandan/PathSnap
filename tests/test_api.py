import numpy as np
from fastapi.testclient import TestClient
from member4_app import api
from tests.test_pipeline import make_trip

class DistanceModel:
    def predict_proba(self,x):
        p=np.clip(np.exp(-x[:,0]/20),.001,.999)
        return np.column_stack([1-p,p])


def test_api_predicts_without_reference_and_rejects_invalid_ids(tmp_path,monkeypatch):
    tr=make_trip(tmp_path)
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'REFERENCES',None)
    monkeypatch.setattr(api,'model',DistanceModel())
    monkeypatch.setattr(api,'metadata',{'selected_model':'test_distance_model','decoder':{'beta':35,'gap_seconds':30}})
    client=TestClient(api.app)
    assert client.get('/trips').json()['trip_ids']==[tr.id]
    response=client.post('/match',json={'trip_id':tr.id})
    assert response.status_code==200
    result=response.json()
    assert len(result['predicted_arc_ids'])==3
    assert result['metrics'] is None
    assert not any('reference' in key or 'label' in key for key in result)
    assert 'Reference route' not in result['map_html']
    assert client.post('/match',json={'trip_id':'../secret'}).status_code==422
    assert client.post('/match',json={'trip_id':'99999999'}).status_code==404


def test_scoring_does_not_change_predictions(tmp_path,monkeypatch):
    tr=make_trip(tmp_path)
    monkeypatch.setattr(api,'DATA',tmp_path);monkeypatch.setattr(api,'model',DistanceModel())
    monkeypatch.setattr(api,'metadata',{'selected_model':'test','decoder':{'beta':35,'gap_seconds':30}})
    monkeypatch.setattr(api,'REFERENCES',None);client=TestClient(api.app)
    first=client.post('/match',json={'trip_id':tr.id}).json()
    folder=tmp_path/tr.id;np.savetxt(folder/f'{tr.id}.route',[0,1,2,3],fmt='%d')
    monkeypatch.setattr(api,'REFERENCES',str(tmp_path))
    second=client.post('/match',json={'trip_id':tr.id}).json()
    assert first['predicted_arc_ids']==second['predicted_arc_ids']
    assert second['metrics'] is not None
    assert not any('labels' in key for key in second)


def test_missing_metadata_returns_unavailable_without_caching(tmp_path, monkeypatch):
    tr = make_trip(tmp_path)
    monkeypatch.setattr(api, 'DATA', tmp_path)
    monkeypatch.setattr(api, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(api, 'model', None)
    monkeypatch.setattr(api, 'metadata', None)
    (tmp_path / 'final_model.joblib').write_bytes(b'placeholder')
    client = TestClient(api.app)
    assert client.get('/health').json()['model_ready'] is False
    assert client.post('/match', json={'trip_id': tr.id}).status_code == 503
    assert api.model is None
    assert not api.lock.locked()
