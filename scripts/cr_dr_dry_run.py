"""CR-DR: run the importer on a CSV in a rolled-back transaction and print the summary (path argument; file read in place)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from apps.web.modules.TradeLog import service
dry = '--commit' not in sys.argv
try:
    r = service.upload_csv(open(sys.argv[1], 'rb').read(), dry_run=dry)
except Exception as e:
    print('upload failed:', type(e).__name__); sys.exit(1)
print(json.dumps(r, indent=1, default=str))
