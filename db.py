import sqlite3
import pandas as pd
import json

from tools import *
from constants import *

def get_tables_and_columns():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables in the database:\n")
    for table in tables:
        table_name = table[0]
        print(f"Table: {table_name}")
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        print(f"  Columns: {', '.join(column_names)}")
    
        # Get number of rows
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"  Rows: {row_count}\n")

    conn.close()

def get_tables_df():
    conn = sqlite3.connect(DB_PATH)
    df_study_patient_data = pd.read_sql_query("SELECT * FROM study_patient_data", conn)
    df_study_instances = pd.read_sql_query("SELECT * FROM study_instances", conn)
    df_study_report = pd.read_sql_query("SELECT * FROM study_report", conn)
    conn.close()
    return df_study_patient_data, df_study_instances, df_study_report  

def group_study_instances(df_study_instances):
    df_study_instances.measurement_data = df_study_instances.measurement_data.apply(json.loads)

    key_columns = ['study_id','file_name']

    non_key_columns = list( set(df_study_instances.columns).difference(key_columns) )

    df_study_instances = df_study_instances.groupby(key_columns).agg({k:list for k in non_key_columns}).reset_index()
    df_study_instances = df_study_instances.rename(columns={
        'measurement_data':'instance_measurement_data',
        'instance_id':'instance_ids',
    })
    return df_study_instances

def group_study_report(df_study_report):
    key_columns = ['study_id']

    non_key_columns = list( set(df_study_report.columns).difference(key_columns) )

    df_study_report = df_study_report.groupby(key_columns).agg({k:list for k in non_key_columns}).reset_index()
    df_study_report = df_study_report.rename(columns={
        'report_id':'report_ids',
        'report_value':'report_values',
    })
    return df_study_report

def run_observations():
    df_study_patient_data, df_study_instances, df_study_report  = get_tables_df()
    
    # Check each study_id has an unique file_name on df_study_instances
    unique_file_name_per_study = df_study_instances.groupby('study_id')['file_name'].nunique()
    multiple_instances = unique_file_name_per_study[unique_file_name_per_study > 1]
    if multiple_instances.empty:
        print("✅ Each study_id has exactly one file_name.")
    else:
        print("❌ Some study_id values have more than one file_name.")
        print("Here they are:")
        print(multiple_instances)
        
    df_study_instances = group_study_instances(df_study_instances)
    df_study_report = group_study_report(df_study_report)
    
    print(f"Is study_id unique in df_study_patient_data? {len(df_study_patient_data.study_id.unique()) == len(df_study_patient_data)}")
    print(f"Is study_id unique in df_study_instances (after grouping)? {len(df_study_instances.study_id.unique()) == len(df_study_instances)}")
    print(f"Is study_id unique in df_study_report (after grouping)? {len(df_study_report.study_id.unique()) == len(df_study_report)}")
    
    df_merged = merge_tables(df_study_patient_data, df_study_instances, df_study_report)
    print(f"N study_id without respective instance_id: {len(df_merged[df_merged.file_name.isna()])}")
    df_merged=df_merged[~df_merged.file_name.isna()]
    print(f"N file_name that doesn't exist: {len(set(df_merged[~df_merged.file_name.apply(does_image_exists)].file_name))}")
    
    print("Available body_parts")
    print(set(x for xs in df_merged.body_parts for x in xs.split(',')))

def merge_tables(df_study_patient_data, df_study_instances, df_study_report):
    df_merged = pd.merge(df_study_patient_data, df_study_instances, on='study_id', how='outer')
    df_merged = pd.merge(df_merged, df_study_report, on='study_id', how='outer')
    return df_merged

def create_dataset():
    df_study_patient_data, df_study_instances, df_study_report  = get_tables_df()
    
    df_study_instances = group_study_instances(df_study_instances)
    df_study_report = group_study_report(df_study_report)
    
    df = merge_tables(df_study_patient_data, df_study_instances, df_study_report)
    
    df=df[~df.file_name.isna()]
    df=df[df.file_name.apply(does_image_exists)]
    
    return df

if __name__ == "__main__":
    run_observations()