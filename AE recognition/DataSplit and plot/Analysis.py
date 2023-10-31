import DataSplit as ds
import SignalProcessing as sp
from DataSplit import rawfile_path
from DataSplit import output_path_sensor1
# from DataSplit import output_path_sensor2
# from DataSplit import output_path_sensor3
# from DataSplit import defect_data_path

"""Use data_analysis to split the files"""
"""Change file names and task_ID accordingly based on TDMS file property"""


def data_analysis():
    ds.data_split(rawfile_path)
    # num_files = 75
    # All the directories concerned parameters are coded in DataSplit.py


def signal_processing():
    sp.signal_processing(output_path_sensor1, 1)
    # sp.signal_processing(output_path_sensor2, 2)
    # sp.signal_processing(output_path_sensor3, 3)


"""comment module if don't need"""

# data_analysis()
signal_processing()
# auto_extraction()
