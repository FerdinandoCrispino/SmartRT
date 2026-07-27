import os
from urllib.parse import quote_plus

class Config:
    """
    Configurações de conexão com o SQL Server via SQLAlchemy + pyodbc.
    """

    DB_SERVER = os.environ.get("DB_SERVER", "localhost")          # ex: "meuservidor.database.windows.net" ou "localhost\\SQLEXPRESS"
    DB_PORT = os.environ.get("DB_PORT", "1433")
    DB_NAME = os.environ.get("DB_NAME", "resultsDB")
    DB_USER = os.environ.get("DB_USER", "usr_GeoPerdas")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "123456")
    DB_DRIVER = os.environ.get("DB_DRIVER", "ODBC Driver 17 for SQL Server")

    # Autenticação do Windows (Trusted Connection). Se True, ignora usuário/senha.
    USE_WINDOWS_AUTH = os.environ.get("USE_WINDOWS_AUTH", "false").lower() == "true"

    @classmethod
    def get_connection_string(cls):
        driver_encoded = quote_plus(cls.DB_DRIVER)

        if cls.USE_WINDOWS_AUTH:
            odbc_str = (
                f"DRIVER={{{cls.DB_DRIVER}}};"
                f"SERVER={cls.DB_SERVER},{cls.DB_PORT};"
                f"DATABASE={cls.DB_NAME};"
                f"Trusted_Connection=yes;"
            )
            params = quote_plus(odbc_str)
            return f"mssql+pyodbc:///?odbc_connect={params}"

        # Autenticação SQL Server (usuário/senha)
        user_encoded = quote_plus(cls.DB_USER)
        password_encoded = quote_plus(cls.DB_PASSWORD)

        return (
            f"mssql+pyodbc://{user_encoded}:{password_encoded}"
            f"@{cls.DB_SERVER}:{cls.DB_PORT}/{cls.DB_NAME}"
            f"?driver={driver_encoded}"
        )

    SQLALCHEMY_DATABASE_URI = None  # preenchido dinamicamente em app.py
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,   # evita erro de conexão "morta"
        "pool_recycle": 3600,
        "fast_executemany": True,
    }
