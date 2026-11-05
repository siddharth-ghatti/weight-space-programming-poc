#!/usr/bin/env python
import json, sys
from pathlib import Path
from wspr.audit import audit_run
print(json.dumps(audit_run(Path(sys.argv[1])), indent=2))
