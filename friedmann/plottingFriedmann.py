import pandas as pd
import matplotlib.pyplot as plt

def read_data():
    """
    Read the Friedmann equation data from a CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the Friedmann equation data.
    """
    # Replace 'friedmann_data.csv' with the path to your CSV file
    df = pd.read_csv('cosmological_distances.csv')
    return df

def plot_distances(df):
    """
    Plot the distances from the Friedmann equation data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the Friedmann equation data.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Big Bang to z Travel Time (Gyr)'], df['Comoving Distance (Mpc)'], label='Comoving')
    plt.plot(df['Big Bang to z Travel Time (Gyr)'], df['Luminosity Distance (Mpc)'], label='Luminosity')
    plt.plot(df['Big Bang to z Travel Time (Gyr)'], df['Proper Distance (Mpc)'], label='Proper')
    # plt.plot(df['Redshift (z)'], df['Angular Distance (Mpc)'], label='Angular')
    plt.title(r'Distances vs Redshift')
    plt.xlabel(r'Redshift ($z$)')
    plt.ylabel(r'Distances ($Mpc$)')
    plt.legend()
    #plt.xscale('log')
    plt.grid()
    plt.show()

data = read_data()
plot_distances(data)
