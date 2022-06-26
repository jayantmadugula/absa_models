'''
This file contains the definition for `DatabaseHandler`,
a class used to easily access data from a SQLite3 database.
'''

from typing import Iterable
import sqlite3
import pandas as pd


class DatabaseHandler():
    def __init__(self, database_path: str) -> None:
        self._db_path = database_path
        self._conn = None

        self._get_db_connection()

    # Public Methods
    def read(
        self,
        table_name: str,
        index_column_name: str = None,
        chunksize: int = None,
        row_indices: range = None,
        columns: Iterable[str] = None,
        retry=True) -> Iterable:
        '''
        Reads data from SQLite database.
        
        If `chunksize` is not `None`, a generator object is returned, with elements
        being `pd.DataFrame` objects.
        Otherwise, this function will attempt to read all of the requested data into memory at once, returning
        a `pd.DataFrame`.
        
        Parameters:
        - `table_name`: name of the table to read from the SQLite3 database

        Optional parameters:
        - `index_column_name`: name of a column in the provided table to use as the index column
        - `chunksize`: the number of rows to include per batch
        - `row_indices`: the indices to read from the database table; if `None`, all rows are read
        - `columns`: a list of columns to read from the database table; if `None`, all columns are read
        - `retry`: will retry the read operation once if set to `True`
        '''
        try:
            table_name = f'"{table_name}"'

            if columns is not None:
                columns = [f'"{c}"' for c in columns]
                if index_column_name not in columns:
                    columns = f'"{index_column_name}", {", ".join(columns)}'
                sql_query = f'SELECT {columns} FROM {table_name}'
                print(sql_query)
            else: 
                sql_query = 'SELECT * FROM {}'.format(table_name)
            if row_indices is not None:
                sql_query += ' WHERE {} IN {}'.format(index_column_name, tuple(row_indices))

            if index_column_name is not None:
                sql_query += f' ORDER BY "{index_column_name}" ASC'

            data = pd.read_sql_query(sql_query, self._conn, index_col=index_column_name, chunksize=chunksize)
        except Exception as e:
            if retry:
                self._get_db_connection()
                self.read(
                    table_name=table_name,
                    index_column_name=index_column_name,
                    chunksize=chunksize,
                    row_indices=row_indices,
                    columns=columns, 
                    retry=False)
            else:
                raise e

        return data
    
    def get_table_length(self, table_name: str):
        '''
        Computes the number of rows in the SQLite3 table.
        '''
        sql_query = 'SELECT COUNT(*) FROM {}'.format(table_name)
        
        c = self._get_db_cursor()
        count = c.execute(sql_query).fetchone()[0]
        self._confirm_operation()
        return count

    def get_longest_document_length(self, table_name: str, data_col_name: str):
        '''
        Finds the length of the document with the most words.
        Assumes words are separated by a single space.
        '''
        # SQL query from:
        # https://stackoverflow.com/questions/41952250/sql-string-counting-words-inside-a-string
        # https://stackoverflow.com/questions/41874972/replacing-variable-length-string-with-some-word

        sql_query = '''
            WITH temp AS
            (
            SELECT  Id,
                    REPLACE(REPLACE(REPLACE({}, ' ', '><'
                                        ), '<>', ''
                                ), '><', ' '
                            ) AS text
            FROM {}
            )
            SELECT id, 
                    length(text) - length(REPLACE(text, ' ', '')) + 1 AS counts 
            FROM temp 
            ORDER BY counts 
            DESC LIMIT 1
        '''.format(data_col_name, table_name)
        
        c = self._get_db_cursor()
        max_len = c.execute(sql_query).fetchone()[-1]
        return max_len

    # Private Methods
    def _get_db_connection(self):
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

    def _confirm_operation(self):
        try:
            self._conn.commit()
        except:
            print('Operation Failed. A new SQLite3 Connection has been generated.')
            self._get_db_connection()

    def _get_db_cursor(self):
        try:
            c = self._conn.cusor()
        except:
            self._get_db_connection()
            c = self._conn.cursor()
        return c