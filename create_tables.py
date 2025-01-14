from csv_to_sql import CSVToSQLite


def create_tables():
    """
    Main entry point. Expects exactly one command-line argument:
    the path to the CSV directory.
    """

    # Create an instance of CSVToSQLite with a desired DB name (optional).
    csv_importer = CSVToSQLite(db_name="./data/catalog.db")

    # Import all CSV files in the given directory.
    csv_importer.import_csvs("./data/")


if __name__ == "__main__":
    create_tables()
