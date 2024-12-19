import os
import csv

def find_files_by_patient_index(directory, patient_index):
    matching_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.dcm'):
            parts = filename.split('_')
            if len(parts) > 1:
                index = parts[1]
                if index == patient_index:
                    matching_files.append(filename)

    return matching_files

def is_valid_patient_index(patient_index):
    directory_path = 'INbreast Release 1.0/ALLDICOMs'
    files = find_files_by_patient_index(directory_path, patient_index)

    if len(files) < 4:
        return False

    found_L_CC = False
    found_L_ML = False
    found_R_CC = False
    found_R_ML = False

    for file in files:
        parts = file.split('_')
        if len(parts) != 6:
            return False
         
        L_or_R = parts[3]
        CC_or_ML = parts[4]

        if L_or_R == 'L' and CC_or_ML == 'CC':
            found_L_CC = True
        elif L_or_R == 'L' and CC_or_ML == 'ML':
            found_L_ML = True
        elif L_or_R == 'R' and CC_or_ML == 'CC':
            found_R_CC = True
        elif L_or_R == 'R' and CC_or_ML == 'ML':
            found_R_ML = True

    return found_L_CC and found_L_ML and found_R_CC and found_R_ML

def extract_patient_indices(directory):
    patient_indices = set()

    for filename in os.listdir(directory):
        if filename.endswith('.dcm'):
            parts = filename.split('_')
            if len(parts) > 1:
                patient_indices.add(parts[1])

    return patient_indices

def main(directory_path, output_csv):
    patient_indices = extract_patient_indices(directory_path)
    valid_patient_indices = []

    existing_indices = set()

    if os.path.exists(output_csv):
        with open(output_csv, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                if row:
                    existing_indices.add(row[0])

    for patient_index in patient_indices:
        if is_valid_patient_index(patient_index) and patient_index not in existing_indices:
            valid_patient_indices.append(patient_index)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not existing_indices:
            writer.writerow(['Patient Index'])
        for index in valid_patient_indices:
            writer.writerow([index])

directory_path = 'INbreast Release 1.0/ALLDICOMs'
output_csv = 'valid_patient_indices.csv'

main(directory_path, output_csv)
print(f"Валидные индексы записаны в '{output_csv}'.")
