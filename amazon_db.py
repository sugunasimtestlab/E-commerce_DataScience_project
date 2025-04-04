# Storing Cleaned Data in MySQL Database

from sqlalchemy import create_engine

USERNAME = "root"
PASSWORD = "Suguna@_1806"
HOST = "localhost"
DATABASE = "amazon_db"  

engine = create_engine(f"mysql+mysqlconnector://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}")

try:
    with engine.connect() as connection:
        print("Connected to MySQL Database successfully!")
except Exception as e:
    print(f"Error: {e}")
