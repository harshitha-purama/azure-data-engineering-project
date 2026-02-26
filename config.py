import os
from dotenv import load_dotenv

load_dotenv(override=True)

SERVER = os.getenv("SQL_SERVER") or os.getenv("SERVER")
DATABASE = os.getenv("SQL_DATABASE") or os.getenv("DATABASE")
USERNAME = os.getenv("SQL_USER") or os.getenv("USERNAME")
PASSWORD = os.getenv("SQL_PASSWORD") or os.getenv("PASSWORD")
DRIVER = os.getenv("SQL_DRIVER") or os.getenv("DRIVER") or "ODBC Driver 18 for SQL Server"