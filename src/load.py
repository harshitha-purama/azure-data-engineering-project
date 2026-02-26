import time
import pyodbc
from pathlib import Path
import sys

try:
    from config import SERVER, DATABASE, USERNAME, PASSWORD, DRIVER
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from config import SERVER, DATABASE, USERNAME, PASSWORD, DRIVER

def get_connection():
    conn_str = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER=tcp:{SERVER},1433;"
        f"DATABASE={DATABASE};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
    )

    last_err = None
    for attempt in range(5):
        try:
            return pyodbc.connect(conn_str, timeout=60)
        except KeyboardInterrupt:
            raise
        except pyodbc.Error as e:
            last_err = e
            time.sleep(min(5 * (attempt + 1), 20))
        except SystemError as e:
            last_err = e
            time.sleep(min(5 * (attempt + 1), 20))
        except Exception as e:
            last_err = e
            time.sleep(min(5 * (attempt + 1), 20))

    raise RuntimeError(
        "Unable to connect to Azure SQL after retries. "
        "If you see SQL error 40613, the database may be temporarily unavailable; "
        "wait a few minutes and retry. "
        "If using Python 3.13 with pyodbc and this persists, try Python 3.12 for better driver stability."
    ) from last_err