import numpy as np
import pandas as pd
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.time import Time
import csv


""" This code is used to generate test datasets to be used in the optimization algorithm. The dataset is generated based on the MASTER-2009 model. """


earth_radius = 6371.0  # in km


file_path_alt = "SMDsimulations/master_results.txt"
file_path_incl = "SMDsimulations/MASTER2_declination_distribution.txt"
file_path_diam = "SMDsimulations/MASTER2_diameter_distribution.txt"

columns_alt = [
    "Altitude", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

columns_incl = [
    "Declination", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

columns_diam = [
    "Diameter", "Expl-Fragm", "Coll_Fragm", "Launch/Mis", "NaK-Drops", "SRM-Slag", "SRM-Dust",
    "Paint Flks", "Ejecta", "MLI", "Cloud 1", "Cloud 2", "Cloud 3", "Cloud 4", "Cloud 5", "Human-made",
    "Meteoroids", "Streams", "Total"
]

data_alt = pd.read_csv(file_path_alt, delim_whitespace=True, skiprows=2, names=columns_alt)
data_incl = pd.read_csv(file_path_incl, delim_whitespace=True, skiprows=2, names=columns_incl)
data_diam = pd.read_csv(file_path_diam, delim_whitespace=True, skiprows=2, names= columns_diam)

total_density = data_incl['Expl-Fragm']
total_sum = total_density.sum()


# Convert the data values
#conversion_factor = 1e-9
#for column in data_alt.columns[1:]:
#    data_alt[column] *= conversion_factor


# Compute probability distributions of debris based on input master model
data_alt['Probability'] = data_alt['Total'] / data_alt['Total'].sum()
data_incl['Probability'] = total_density/total_sum
data_diam['Probability'] = data_diam['Total'] / data_diam['Total'].sum()

dataset_sizes = np.append(np.arange(100,1001,100), np.arange(2000,10001,1000))
print(dataset_sizes)
for size in dataset_sizes:

    total_particles = size  # Total number of debris particles (to simulate)

    # Select random orbit altitudes, inclinations and diameters based on the computed probabilities
    altitudes = np.random.choice(data_alt['Altitude'], size=total_particles, p=data_alt['Probability'])
    latitudes = np.random.choice(data_incl['Declination'], size=total_particles, p=data_incl['Probability'])
    latitudes += 90
    diameters = np.random.choice(data_diam['Diameter'], size=total_particles, p=data_diam['Probability'])

    # Choose remaining orbital elements randomly
    eccentricities = np.random.uniform(-0.05, 0.05, total_particles)
    r_ascension = np.random.uniform(0,360, total_particles)
    arg_periapsis = np.random.uniform(0, 360, total_particles)
    true_anomaly = np.random.uniform(0, 360, total_particles)
    radii = (altitudes+earth_radius)


    dataset = {
        "radii": radii,
        "latitudes": latitudes,
        "eccentricities": eccentricities,
        "r_ascension": r_ascension,
        "arg_periapsis": arg_periapsis,
        "true_anomaly": true_anomaly,
        "diameters": diameters,
        "altitudes": altitudes,
    }


    # Save to csv file
    filename = f'Optimization/test_datasets/{size}debris.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=dataset.keys())
        writer.writeheader()
        writer.writerows([dict(zip(dataset.keys(), row)) for row in zip(*dataset.values())])

    print(f"List saved successfully to '{filename}'")



    if True:
        ######## to initialize the debris from test datasets ########


        filename = f'Optimization/test_datasets/{size}debris.csv'
        # Read CSV into a dictionary
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            dataset_parsed = {key: [] for key in reader.fieldnames}  # Initialize an empty dictionary with column headers
            for row in reader:
                for key in row:
                    dataset_parsed[key].append(row[key])  # Append each value to the corresponding list

        print(f"Dictionary reconstructed from '{filename}':")
        #print(data)


        radii_parsed = np.array(dataset_parsed['radii'], dtype=np.float64)
        latitudes_parsed = np.array(dataset_parsed['latitudes'], dtype=np.float64)
        eccentricities_parsed = np.array(dataset_parsed['eccentricities'], dtype=np.float64)
        r_ascension_parsed = np.array(dataset_parsed['r_ascension'], dtype=np.float64)
        arg_periapsis_parsed = np.array(dataset_parsed['arg_periapsis'], dtype=np.float64)
        true_anomaly_parsed = np.array(dataset_parsed['true_anomaly'], dtype=np.float64)
        diameters_parsed = np.array(dataset_parsed['diameters'], dtype=np.float64)
        altitudes_parsed = np.array(dataset_parsed['altitudes'], dtype=np.float64)

        parsed_difference = [radii - radii_parsed,
                            latitudes - latitudes_parsed,
                            eccentricities - eccentricities_parsed,
                            r_ascension - r_ascension_parsed,
                            arg_periapsis - arg_periapsis_parsed,
                            true_anomaly - true_anomaly_parsed,
                            diameters - diameters_parsed,
                            altitudes - altitudes_parsed]

    print("Total difference: ", sum(sum(parsed_difference)))
    if sum(sum(parsed_difference)) == 0:
        print(f"Parsed data for {size} debris dataset matches original data.")


