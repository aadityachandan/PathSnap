"""Folium maps: reference layer allowed only in development reports."""
import folium
import numpy as np
from .decoder import geometry


def make_map(trip,pieces,reference=None):
    center=trip.track[:,[1,0]].mean(axis=0).tolist()
    m=folium.Map(location=center,zoom_start=14,tiles='OpenStreetMap')
    roads=folium.FeatureGroup(name='Nearby road network',show=False)
    lo=trip.gps.min(axis=0)-150;hi=trip.gps.max(axis=0)+150
    a=trip.xy[trip.arcs[:,0]];b=trip.xy[trip.arcs[:,1]]
    visible=np.flatnonzero(np.all(np.maximum(a,b)>=lo,axis=1)&np.all(np.minimum(a,b)<=hi,axis=1))
    for arc in visible[:12000]:folium.PolyLine(trip.nodes[trip.arcs[arc]][:,::-1].tolist(),color='#8799ad',weight=1,opacity=.45).add_to(roads)
    roads.add_to(m)
    gps=folium.FeatureGroup(name='Observed GPS (blue)')
    for lat,lon in trip.track[:,[1,0]][::max(1,len(trip.track)//1500)]:
        folium.CircleMarker([float(lat),float(lon)],radius=2,color='#337dff',fill=True,fill_opacity=.65).add_to(gps)
    gps.add_to(m)
    if reference is not None:
        layer=folium.FeatureGroup(name='Reference route - development only (orange)')
        for line in geometry(trip,[reference.tolist()]):folium.PolyLine(line,color='#f59e0b',weight=6,opacity=.6).add_to(layer)
        layer.add_to(m)
    layer=folium.FeatureGroup(name='Predicted route (green)')
    for line in geometry(trip,pieces):folium.PolyLine(line,color='#10b981',weight=4).add_to(layer)
    layer.add_to(m);folium.LayerControl().add_to(m)
    m.fit_bounds([trip.track[:,[1,0]].min(axis=0).tolist(),trip.track[:,[1,0]].max(axis=0).tolist()])
    return m
