import os


def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if "Signal" in filename:  # Check if the file name matches the pattern
            parts = filename.split(" ")
            start_time = parts[1]
            end_time = parts[3]

            if "." in start_time and len(start_time.split(".")[1]) == 1:
                start_time = start_time.split(".")[0] + "." + start_time.split(".")[1] + "0"

            if "." in end_time and len(end_time.split(".")[1]) == 1:
                end_time = end_time.split(".")[0] + "." + end_time.split(".")[1] + "0"

            # Check if filename contains "-spectrum"
            if "-spectrum" in filename:
                new_filename = f"Signal {start_time} to {end_time} sec-spectrum.png"
            else:
                new_filename = f"Signal {start_time} to {end_time} sec.png"

            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


directory_path = "/Volumes/Extreme SSD/2023-09-08/train/2nd defect at web-40db/plot/"  # Replace this with the path
# to your directory
rename_files_in_directory(directory_path)

