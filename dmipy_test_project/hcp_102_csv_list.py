import os
import csv

base_path = r'/nfs/masi/nathv/miccai_2020_hcp_100'
base_path = os.path.normpath(base_path)
subj_list = os.listdir(base_path)

with open('hcp_102.csv', mode='w') as hcp_csv_file:
    csv_writer = csv.writer(hcp_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for each_subj in subj_list:
        csv_writer.writerow([str(each_subj)])

