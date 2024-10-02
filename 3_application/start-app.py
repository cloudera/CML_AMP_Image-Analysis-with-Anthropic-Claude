import subprocess
import os

## start streamlit app
print(subprocess.run(["streamlit run ./3_application/app.py --server.port $CDSW_APP_PORT --server.address 127.0.0.1"], shell=True))
