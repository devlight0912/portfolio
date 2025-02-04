import sqlite3
from typing import Optional, Dict, Any, Union, List
import psycopg2
from psycopg2 import OperationalError, sql
import pandas as pd
import logging
import json
from contextlib import contextmanager
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_schema_cache: Dict[str, Any] = {}

def create_connection(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Optional[Union[sqlite3.Connection, psycopg2.extensions.connection]]:
    """
    Create a database connection for either SQLite or PostgreSQL.
    """
    try:
        if db_type.lower() == 'postgresql':
            conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host
            )
            logger.info("Connected to PostgreSQL database.")
        elif db_type.lower() == 'sqlite':
            conn = sqlite3.connect(db_name)
            logger.info("Connected to SQLite database.")
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return None
        return conn
    except OperationalError as e:
        logger.error(f"Operational error while connecting to the database: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error while connecting to the database: {e}")
    return None

@contextmanager
def get_connection(db_name: str, db_type: str, host: Optional[str]=None,
                   user: Optional[str]=None, password: Optional[str]=None):
    """
    Context manager for database connections.
    """
    conn = create_connection(db_name, db_type, host, user, password)
    try:
        yield conn
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

def query_database(
    query: str,
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> pd.DataFrame:
    """
    Execute an SQL query and return the results as a DataFrame.
    """
    with get_connection(db_name, db_type, host, user, password) as conn:
        if conn is None:
            logger.error("Database connection failed. Returning empty DataFrame.")
            return pd.DataFrame()
        try:
            df = pd.read_sql_query(query, conn)
            logger.info("Query executed successfully.")
            return df
        except Exception as e:
            logger.exception(f"Unexpected error executing query: {e}")
            return pd.DataFrame()

# --- Abstract Schema Extractor Classes ---

class SchemaExtractor(ABC):
    """
    Abstract base class for schema extraction.
    """
    def __init__(self, connection):
        self.conn = connection

    @abstractmethod
    def get_tables(self) -> List[str]:
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        pass

class SQLiteSchemaExtractor(SchemaExtractor):
    """
    Extract schema information from a SQLite database.
    """
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        table_info = get_sqlite_table_info(cursor, table_name)
        # Optionally, add additional SQLite-specific extraction here (e.g., triggers, check constraints)
        return table_info

class PostgreSQLSchemaExtractor(SchemaExtractor):
    """
    Extract schema information from a PostgreSQL database.
    """
    def get_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cursor.fetchall()]
        return tables

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        table_info = get_postgresql_table_info(cursor, table_name)
        # Optionally, add additional PostgreSQL-specific extraction here (e.g., triggers, unique constraints)
        return table_info

# --- SQLite Schema Extraction ---

def get_sqlite_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieve schema information for a given SQLite table.
    """
    table_info = {
        'columns': {},
        'foreign_keys': [],
        'indexes': [],
        'sample_data': [],
        'primary_keys': [],
        # Placeholder for additional constraints/triggers if needed:
        'constraints': [],
        'triggers': []
    }

    # Get column information
    cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
    columns = cursor.fetchall()
    for col in columns:
        col_name = col[1]
        table_info['columns'][col_name] = {
            'type': col[2],
            'nullable': not col[3],
            'default': col[4],
            'primary_key': bool(col[5])
        }
        if col[5]:
            table_info['primary_keys'].append(col_name)

    # Get foreign key constraints
    cursor.execute(f"PRAGMA foreign_key_list(\"{table_name}\");")
    fkeys = cursor.fetchall()
    for fk in fkeys:
        table_info['foreign_keys'].append({
            'from_column': fk[3],
            'to_table': fk[2],
            'to_column': fk[4],
            'on_update': fk[5],
            'on_delete': fk[6]
        })

    # Get indexes
    cursor.execute(f"PRAGMA index_list(\"{table_name}\");")
    indexes = cursor.fetchall()
    for idx in indexes:
        cursor.execute(f"PRAGMA index_info(\"{idx[1]}\");")
        index_columns = cursor.fetchall()
        table_info['indexes'].append({
            'name': idx[1],
            'unique': bool(idx[2]),
            'columns': [col[2] for col in index_columns]
        })

    # Get sample data
    cursor.execute(f"SELECT * FROM \"{table_name}\" LIMIT 5;")
    sample_data = cursor.fetchall()
    if sample_data:
        column_names = [desc[0] for desc in cursor.description]
        table_info['sample_data'] = [
            dict(zip(column_names, row)) for row in sample_data
        ]

    return table_info

# --- PostgreSQL Schema Extraction ---

def get_postgresql_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieve schema information for a given PostgreSQL table.
    """
    table_info = {
        'columns': {},
        'foreign_keys': [],
        'indexes': [],
        'sample_data': [],
        'primary_keys': [],
        # Placeholder for additional constraints/triggers if needed:
        'constraints': [],
        'triggers': []
    }

    # Get column information
    cursor.execute("""
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s;
    """, [table_name])
    columns = cursor.fetchall()
    for col in columns:
        table_info['columns'][col[0]] = {
            'type': col[1],
            'nullable': col[2] == 'YES',
            'default': col[3],
            'max_length': col[4],
            'primary_key': False  # Will update below
        }

    # Get primary key information
    cursor.execute("""
        SELECT kcu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY';
    """, [table_name])
    pk_columns = [row[0] for row in cursor.fetchall()]
    for col in pk_columns:
        if col in table_info['columns']:
            table_info['columns'][col]['primary_key'] = True
            table_info['primary_keys'].append(col)

    # Get foreign key information
    cursor.execute("""
        SELECT
            kcu.column_name AS from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column,
            rc.update_rule AS on_update,
            rc.delete_rule AS on_delete
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON tc.constraint_name = ccu.constraint_name
        JOIN information_schema.referential_constraints AS rc
          ON tc.constraint_name = rc.constraint_name
        WHERE tc.table_name = %s AND tc.constraint_type = 'FOREIGN KEY';
    """, [table_name])
    fkeys = cursor.fetchall()
    for fk in fkeys:
        table_info['foreign_keys'].append({
            'from_column': fk[0],
            'to_table': fk[1],
            'to_column': fk[2],
            'on_update': fk[3],
            'on_delete': fk[4]
        })

    # Get indexes
    cursor.execute("""
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s;
    """, [table_name])
    indexes = cursor.fetchall()
    for idx_name, idx_def in indexes:
        # Extract index columns from definition (this is a simple heuristic)
        try:
            idx_columns_part = idx_def.split('(')[1].rstrip(')')
            idx_columns = [col.strip() for col in idx_columns_part.split(',')]
        except Exception as e:
            logger.warning(f"Failed to extract index columns from {idx_def}: {e}")
            idx_columns = []
        is_unique = 'UNIQUE' in idx_def.upper()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': is_unique,
            'columns': idx_columns
        })

    # Get sample data using safe identifier formatting
    cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 5;").format(sql.Identifier(table_name)))
    sample_data = cursor.fetchall()
    if sample_data:
        column_names = [desc[0] for desc in cursor.description]
        table_info['sample_data'] = [
            dict(zip(column_names, row)) for row in sample_data
        ]

    return table_info

# --- Generate Structured JSON Schema Representation ---

def generate_json_schema(table_name: str, table_info: Dict[str, Any]) -> str:
    """
    Generates a JSON representation of the table schema including attributes,
    relationships, indexes, constraints, and triggers.
    """
    schema = {
        "object": table_name,
        "attributes": [],
        "relationships": [],
        "indexes": table_info.get('indexes', []),
        "constraints": table_info.get('constraints', []),
        "triggers": table_info.get('triggers', []),
        "sample_data": table_info.get('sample_data', [])
    }

    for col_name, details in table_info['columns'].items():
        attribute = {
            "name": col_name,
            "type": details.get('type'),
            "primary_key": details.get('primary_key', False),
            "nullable": details.get('nullable', True),
            "default": details.get('default'),
            "max_length": details.get('max_length')
        }
        schema["attributes"].append(attribute)

    for fk in table_info.get('foreign_keys', []):
        relationship = {
            "from_column": fk['from_column'],
            "to_table": fk['to_table'],
            "to_column": fk['to_column'],
            "on_update": fk['on_update'],
            "on_delete": fk['on_delete']
        }
        schema["relationships"].append(relationship)

    return json.dumps(schema, indent=2)

# --- Unified Schema Retrieval ---

def get_all_schemas(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve all table schemas from the specified database as a dictionary where
    keys are table names and values are the structured schema representations.
    """
    # Build a cache key if needed (example: f"{db_type}:{db_name}:{host or 'localhost'}")
    cache_key = f"{db_type}:{db_name}:{host or 'localhost'}"
    if cache_key in _schema_cache:
        logger.info("Returning cached schemas.")
        return _schema_cache[cache_key]

    schemas: Dict[str, Dict[str, Any]] = {}
    with get_connection(db_name, db_type, host, user, password) as conn:
        if conn is None:
            logger.error("Database connection failed. Returning empty schemas.")
            return {}

        # Choose the appropriate extractor
        if db_type.lower() == 'sqlite':
            extractor: SchemaExtractor = SQLiteSchemaExtractor(conn)
        elif db_type.lower() == 'postgresql':
            extractor = PostgreSQLSchemaExtractor(conn)
        else:
            logger.error(f"Unsupported database type: {db_type}")
            return {}

        # Retrieve schemas for each table
        for table in extractor.get_tables():
            table_info = extractor.get_table_info(table)
            # Generate a structured JSON schema representation
            table_info['ora_representation'] = generate_json_schema(table, table_info)
            schemas[table] = table_info

    # Cache the result for future calls
    _schema_cache[cache_key] = schemas
    return schemas
