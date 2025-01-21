import sqlite3
from typing import Optional, Dict, Any, Union
import psycopg2
from psycopg2 import OperationalError, sql
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_schema_cache = {}

def create_connection(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Optional[Union[sqlite3.Connection, psycopg2.extensions.connection]]:
    """
    Create or connect to a database.

    Parameters:
    - db_name (str): Name of the database.
    - db_type (str): Type of the database ('sqlite' or 'postgresql').
    - host (Optional[str]): Host address (for PostgreSQL).
    - user (Optional[str]): Username (for PostgreSQL).
    - password (Optional[str]): Password (for PostgreSQL).

    Returns:
    - Optional[Connection]: Database connection object or None if connection fails.
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

    Parameters:
    - query (str): The SQL query to execute.
    - db_name (str): Name of the database.
    - db_type (str): Type of the database ('sqlite' or 'postgresql').
    - host (Optional[str]): Host address (for PostgreSQL).
    - user (Optional[str]): Username (for PostgreSQL).
    - password (Optional[str]): Password (for PostgreSQL).

    Returns:
    - pd.DataFrame: Query results.
    """
    conn = create_connection(db_name, db_type, host, user, password)
    if conn is None:
        logger.error("Database connection failed. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        df = pd.read_sql_query(query, conn)
        logger.info("Query executed successfully.")
        return df
    except psycopg2.OperationalError as e:
        logger.error(f"PostgreSQL operational error: {e}")
        return pd.DataFrame()
    except sqlite3.DatabaseError as e:
        logger.error(f"SQLite database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.exception(f"Unexpected error executing query: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
        logger.info("Database connection closed.")

def get_all_schemas(
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve schema information from the database.
    Checks cache first; if not found, fetches from DB and stores in cache.
    """
    cache_key = (db_name, db_type, host, user, password)
    if cache_key in _schema_cache:
        logger.info("Schema retrieved from cache.")
        return _schema_cache[cache_key]

    conn = create_connection(db_name, db_type, host, user, password)
    if conn is None:
        logger.error("Database connection failed. Returning empty schemas.")
        return {}

    schemas = {}
    try:
        cursor = conn.cursor()

        if db_type.lower() == 'sqlite':
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]

            for table_name in tables:
                table_info = get_sqlite_table_info(cursor, table_name)
                table_info['ora_representation'] = generate_ora_representation(table_name, table_info)
                schemas[table_name] = table_info

        elif db_type.lower() == 'postgresql':
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            tables = [table[0] for table in cursor.fetchall()]

            for table_name in tables:
                table_info = get_postgresql_table_info(cursor, table_name)
                table_info['ora_representation'] = generate_ora_representation(table_name, table_info)
                schemas[table_name] = table_info

        else:
            logger.error(f"Unsupported database type: {db_type}")
            return {}

        _schema_cache[cache_key] = schemas
        logger.info("Schema information retrieved and cached successfully.")
        return schemas

    except Exception as e:
        logger.exception(f"Error retrieving schema information: {e}")
        return {}

    finally:
        conn.close()
        logger.info("Database connection closed.")

def get_sqlite_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieve table schema information for SQLite.

    Parameters:
    - cursor: SQLite database cursor.
    - table_name (str): The name of the table.

    Returns:
    - Dict[str, Any]: Dictionary containing table schema information.
    """
    table_info = {'columns': {}, 'foreign_keys': [], 'indexes': [], 'sample_data': [], 'primary_keys': []}

    # Get column information
    cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
    columns = cursor.fetchall()
    for col in columns:
        table_info['columns'][col[1]] = {
            'type': col[2],
            'nullable': not col[3],
            'primary_key': bool(col[5]),
            'default': col[4]
        }
        if col[5]:  # If it's a primary key
            table_info['primary_keys'].append(col[1])

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
        column_names = [description[0] for description in cursor.description]
        table_info['sample_data'] = [
            dict(zip(column_names, row)) for row in sample_data
        ]

    return table_info

def get_postgresql_table_info(cursor, table_name: str) -> Dict[str, Any]:
    """
    Retrieve table schema information for PostgreSQL.

    Parameters:
    - cursor: PostgreSQL database cursor.
    - table_name (str): The name of the table.

    Returns:
    - Dict[str, Any]: Dictionary containing table schema information.
    """
    table_info = {'columns': {}, 'foreign_keys': [], 'indexes': [], 'sample_data': [], 'primary_keys': []}

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
            'primary_key': False  # Will be updated later if it's a primary key
        }

    # Get primary key information
    cursor.execute("""
        SELECT kcu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY';
    """, [table_name])
    pk_columns = [col[0] for col in cursor.fetchall()]
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
        # Extract columns from index definition
        idx_columns = idx_def.split('(')[1].rstrip(')').split(', ')
        is_unique = 'UNIQUE' in idx_def.upper()
        table_info['indexes'].append({
            'name': idx_name,
            'unique': is_unique,
            'columns': idx_columns
        })

    # Get sample data
    cursor.execute(sql.SQL("SELECT * FROM {} LIMIT 5;").format(
        sql.Identifier(table_name)))
    sample_data = cursor.fetchall()
    if sample_data:
        column_names = [desc[0] for desc in cursor.description]
        table_info['sample_data'] = [
            dict(zip(column_names, row)) for row in sample_data
        ]

    return table_info

def generate_ora_representation(table_name: str, table_info: Dict[str, Any]) -> str:
    """
    Generates an Object-Relation-Attribute (ORA) representation of the table schema.

    Parameters:
    - table_name (str): The name of the table.
    - table_info (Dict[str, Any]): The schema information for the table.

    Returns:
    - str: The ORA representation of the table.
    """
    ora_parts = [f"Object: {table_name}"]
    attributes = []
    relationships = []

    for col_name, col_details in table_info['columns'].items():
        attr_desc = f"  Attribute: {col_name} (Type: {col_details['type']}"
        if col_details.get('primary_key'):
            attr_desc += ", Primary Key"
        if not col_details['nullable']:
            attr_desc += ", Not Null"
        if col_details.get('default') is not None:
            attr_desc += f", Default: {col_details['default']}"
        attr_desc += ")"
        attributes.append(attr_desc)

    for fk in table_info['foreign_keys']:
        relationships.append(
            f"  Relationship: References {fk['to_table']}.{fk['to_column']} via {fk['from_column']} (On Update: {fk['on_update']}, On Delete: {fk['on_delete']})"
        )

    ora_parts.extend(attributes)
    ora_parts.extend(relationships)

    return "\n".join(ora_parts)
