"""Laptop 1 launcher. Run from any directory; supply the trip-folder parent."""
import argparse,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
p=argparse.ArgumentParser(description='Start the PathSnap model API on Laptop 1')
p.add_argument('--trips',required=True,help='Directory containing trip ID folders')
p.add_argument('--references',help='Optional directory containing server-side reference trip folders')
p.add_argument('--port',type=int,default=8000)
a=p.parse_args()
if not Path(a.trips).is_dir():p.error('--trips must be an existing directory')
os.environ['PATHSNAP_TRIPS']=str(Path(a.trips).resolve())
if a.references:os.environ['PATHSNAP_REFERENCES']=str(Path(a.references).resolve())
else:os.environ.pop('PATHSNAP_REFERENCES',None)
import uvicorn
print(f'Laptop 1 API: http://127.0.0.1:{a.port}/docs',flush=True)
print(f'On Laptop 2, use http://<Laptop-1-LAN-IP>:{a.port}',flush=True)
uvicorn.run('member4_app.api:app',host='0.0.0.0',port=a.port)
