import psycopg2
import pandas as pd

class conn_db:
    def __init__(self, database, user, password, host='localhost', port='5432'):
        """Initialize the PostgreSQL connection parameters."""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish a connection to the TellCo_db database."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            print("Connected to TellCo_db database successfully.")
        except (Exception, psycopg2.Error) as error:
            print(f"Error connecting to TellCo_db database: {error}")

    def disconnect(self):
        """Close the TellCo_db database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("TellCo_db connection closed.")

    def execute_query(self, query):
        """Execute a query in the PostgreSQL database."""
        try:
            if self.connection is None:
                self.connect()  # Ensure the connection is established before executing the query
            self.cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully.")
        except (Exception, psycopg2.Error) as error:
            print(f"Error executing query: {error}")

    def fetch_data(self, query):
        """Fetch data from TellCo_db database and return it as a pandas DataFrame."""
        try:
            if self.connection is None:
                self.connect()  # Ensure the connection is established before fetching data
            self.cursor.execute(query)
            data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=column_names)
            return df
        except (Exception, psycopg2.Error) as error:
            print(f"Error fetching data: {error}")
            return None
