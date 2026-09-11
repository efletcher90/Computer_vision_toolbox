import pathlib as pl
import os
import csv

path = pl.Path(r'/data/raw/test_images/gt_counts')

def run():
    file_list = [file for file in os.listdir(path) if file.lower().endswith('.png')]

    counts = []
    tif_files = []
    files = []
    for file_name in file_list:
        stem = file_name.removesuffix('cells.png')
        filename_split = stem.split('_')
        last_item = filename_split[-1]
        counts.append(last_item)
        files.append(file_name)
        tif_shrunk = file_name[:-19] + ".tif"
        tif_files.append(tif_shrunk)

    count_dict = [{"file name": file, "count": count} for file, count in zip(files, counts)]
    tif_dict = [{"file name": file, "count": count} for file, count in zip(tif_files, counts)]

    return count_dict, tif_dict

def print_to_excel(file_dict):
    gt_counts = path / 'gt_cell_counts_tifs.csv'
    fields = ['file name', 'count']
    with open(gt_counts, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(file_dict)

if __name__ == '__main__':
    _, tif_counts = run()
    print_to_excel(tif_counts)
