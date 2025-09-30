from dotenv import load_dotenv
import pandas as pd

def run():
    load_dotenv()
    print("Template OK — environment loaded.")
    print("Hello world!")
    print(pd.read_csv("src/ingest/datalab_export_2025-09-29 13_50_15.csv").head())

if __name__ == "__main__":
    run()
