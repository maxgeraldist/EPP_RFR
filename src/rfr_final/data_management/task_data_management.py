"""Tasks for managing the data."""

import pandas as pd
import pytask
import os
import sys
current_file_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(current_file_dir)
sys.path.append(src_dir)
from clean_data import task_process_data



@pytask.mark.depends_on(os.path.join(src_dir, "..", "data", "weekly_prepared_26_11_2017.xlsx"))
@pytask.mark.produces(os.path.join(src_dir, "..", "data", "data_clean.xlsx"))
def task_clean_data(depends_on, produces):
    # Load raw data from XLSX file
    df = pd.read_excel(depends_on, sheet_name="JTI_weekly_prepared_26_11_2017")

    # Process data using task_process_data function
    df = task_process_data(df)

    # Save processed data to Excel file
    df.to_excel(produces)   
    