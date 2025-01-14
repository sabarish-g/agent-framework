from pathlib import Path
import sqlite3
import pandas as pd
import pdb


class CSVToSQLite:
    """
    A class to read multiple CSV files from a directory and import each one
    as a separate table into a single SQLite database file.
    """

    def __init__(self, db_name="catalog.db"):
        """
        :param db_name: Name of the SQLite database file (default: catalog.db).
        """
        self.db_name = db_name

    def import_csvs(self, csv_directory):
        """
        Connect to the SQLite database (creates it if not found).
        For each .csv file in 'csv_directory', create/update a table named
        after the sanitized file name.
        """
        # Convert csv_directory to a Path object
        csv_path = Path(csv_directory)
        
        conn = sqlite3.connect(self.db_name)

        # Iterate over all CSV files in the directory using pathlib
        for file_path in csv_path.iterdir():
            if file_path.suffix.lower() == ".csv":
                # file_path.stem is the filename without extension
                base_name = file_path.stem.strip().lower().replace("-","_").replace(" ", "_")
                pdb.set_trace()

                # Read CSV into a pandas DataFrame
                df = pd.read_csv(file_path)

                # Write DataFrame to SQLite
                df.to_sql(base_name, conn, if_exists="replace", index=False)

                print(f"Imported '{file_path.name}' into table '{base_name}' in '{self.db_name}'.")

        conn.close()
        print(f"All CSVs have been imported into '{self.db_name}'.")
