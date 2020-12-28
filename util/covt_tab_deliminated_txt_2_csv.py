import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_name",
    type=str,
    required=True
)
args = parser.parse_args()
file_name = args.file_name

# read tab-delimited file
with open(f'{file_name}.txt', 'r') as fin:
    cr = csv.reader(fin, delimiter='\t')
    filecontents = [list(filter(None, line)) for line in cr]

print(filecontents)

# write comma-delimited file (comma is the default delimiter)
with open(f'{file_name}.csv', 'w', newline="") as fou:
    cw = csv.writer(fou, quotechar='', quoting=csv.QUOTE_NONE)
    cw.writerow(['LBA',
                 'SizeInSector',
                 'SizeInByte',
                 'IOType',
                 'Timestamp',
                 'ProcessStartTime',
                 'HWSubmittedTime',
                 'ProcessFinishTime'])
    cw.writerows(filecontents)

