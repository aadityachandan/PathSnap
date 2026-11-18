import numpy as np
from pathlib import Path
from pathsnap.data import load_trip,candidates,project
from pathsnap.reference import align_reference
from pathsnap.decoder import decode,geometry


def make_trip(tmp_path,disconnected=False):
    d=tmp_path/'00000001';d.mkdir()
    nodes=np.array([[8,49],[8.001,49],[8.002,49],[8.003,49],[8.004,49]])
    arcs=np.array([[0,1],[1,2],[2,3],[3,4],[1,0],[2,1],[3,2],[4,3]])
    if disconnected:arcs=np.array([[0,1],[3,4]])
    track=np.array([[8.0002,49,1],[8.0008,49,2],[8.0015,49,3]])
    for ext,data in [('nodes',nodes),('arcs',arcs),('track',track)]:np.savetxt(d/f'{d.name}.{ext}',data,fmt='%d' if ext=='arcs' else '%.8f')
    return load_trip(d)


def test_candidates_are_exact_nearest_segments_without_route_file(tmp_path):
    tr=make_trip(tmp_path);ids,f=candidates(tr)
    for t,p in enumerate(tr.gps):
        dist,_=project(p,tr.xy[tr.arcs[:,0]],tr.xy[tr.arcs[:,1]])
        expected=np.lexsort((np.arange(len(dist)),dist))[:ids.shape[1]]
        assert np.array_equal(ids[t],expected)
    assert np.isfinite(f).all()


def test_reference_alignment_is_ordered(tmp_path):
    tr=make_trip(tmp_path);labels,dist=align_reference(tr,np.array([0,1,2,3]))
    assert np.all(np.diff(labels)>=0)
    assert np.max(dist)<.001


def test_decoder_returns_valid_connected_pieces(tmp_path):
    tr=make_trip(tmp_path);ids,f=candidates(tr);p=np.exp(-f[:,:,0]/20)*np.exp(f[:,:,1])
    pred,pieces,breaks=decode(tr,ids,f,p)
    assert len(pred)==len(tr.track)
    assert all(pred[t] in ids[t] for t in range(len(pred)))
    for piece in pieces:
        for left,right in zip(piece,piece[1:]):assert tr.arcs[left,1]==tr.arcs[right,0]
    assert len(geometry(tr,pieces))>=1


def test_disconnected_roads_and_large_gap_restart(tmp_path):
    tr=make_trip(tmp_path,True)
    tr.track[1,2]=100;tr.track[2,2]=101
    ids=np.array([[0],[1],[1]],dtype=int)
    f=np.zeros((3,1,8));f[:,:,3]=.5
    pred,pieces,breaks=decode(tr,ids,f,np.ones((3,1)))
    assert 1 in breaks and len(pieces)==2
    assert pred.tolist()==[0,1,1]


def test_invalid_timestamp_rejected(tmp_path):
    tr=make_trip(tmp_path);folder=tmp_path/tr.id
    tr.track[1,2]=tr.track[0,2];np.savetxt(folder/f'{tr.id}.track',tr.track)
    import pytest
    with pytest.raises(ValueError,match='timestamps'):load_trip(folder)
