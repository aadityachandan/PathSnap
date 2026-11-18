"""Laptop 2 launcher; Python standard library only."""
import argparse,functools
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8501);a=p.parse_args()
handler=functools.partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).resolve().parent/'frontend'))
print(f'Open http://localhost:{a.port} in your browser. Enter Laptop 1 API address.',flush=True)
try:ThreadingHTTPServer(('0.0.0.0',a.port),handler).serve_forever()
except KeyboardInterrupt:print('UI stopped.')
