import os
from urllib.parse import urlparse

from optuna_dashboard import run_server

from dotenv import load_dotenv
load_dotenv()

optuna_storage = os.getenv("OPTUNA_STORAGE")
optuna_uri = os.getenv("OPTUNA_URI")

if optuna_uri:
    parsed_uri = urlparse(optuna_uri)
    host = parsed_uri.hostname
    port = parsed_uri.port
    run_server(storage=optuna_storage, host=host, port=port)
