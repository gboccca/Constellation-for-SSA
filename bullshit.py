import csv

if True:
    ######## to initialize the debris from test datasets ########

    total_particles = 1000
    filename = f'Optimization/test_datasets/{total_particles}debris.csv'
    # Read CSV into a dictionary
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        data = {key: [] for key in reader.fieldnames}  # Initialize an empty dictionary with column headers
        for row in reader:
            for key in row:
                data[key].append(row[key])  # Append each value to the corresponding list

    print(f"Dictionary reconstructed from '{filename}':")
    print(data)


