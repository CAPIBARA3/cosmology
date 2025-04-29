import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from codecarbon import EmissionsTracker


# Define the updated CSV data as a string
data = """Direct/Indirect,Type,First Author,et al,Year,Datasets,Value,Lower,Upper,ArXiv
Indirect,CMB with Planck,Pogosian,Y,2020,eBOSS+Planck,69.6,1.8,1.8,2009.08455
Indirect,CMB with Planck,Aghanim,Y,2020,Planck18,67.27,0.6,0.6,1807.06209
Indirect,CMB with Planck,Aghanim,Y,2020,Planck18+CMB lensing,67.36,0.54,0.54,1807.06209
Indirect,CMB with Planck,Ade,Y,2016,Planck15,67.27,0.66,0.66,1502.01589
Indirect,CMB without Planck,Aiola,Y,2020,ACT,67.9,1.5,1.5,2007.07288
Indirect,CMB without Planck,Aiola,Y,2020,ACT+WMAP9,67.6,1.1,1.1,2007.07288
Indirect,CMB without Planck,Zhang,Y,2019,WMAP+BAO,68.36,0.52,0.53,1812.01877
Indirect,CMB without Planck,Henning,Y,2018,SPT,71.3,2.1,2.1,1707.09353
Indirect,CMB without Planck,Hinshaw,Y,2013,WMAP9,70,2.2,2.2,1212.5226
Direct,Cepheids-SNIa,Riess,Y,2020,R20,73.2,1.3,1.3,2012.08534
Direct,Cepheids-SNIa,Breuval,Y,2020,SNIa,72.8,2.7,2.7,2006.08763
Direct,Cepheids-SNIa,Riess,Y,2019,R19,74,1.4,1.4,1903.07603
Direct,Cepheids-SNIa,Camarena,Y,2019,SNIa,75.4,1.7,1.7,1906.11814
Direct,Cepheids-SNIa,Burns,Y,2018,SNIa,73.2,2.3,2.3,1809.06381
Direct,Cepheids-SNIa,Dhawan,Y,2017,NIR,72.8,3.1,3.1,1707.00715
Direct,Cepheids-SNIa,Follin,Y,2017,SNIa,73.3,1.7,1.7,1707.01175
Direct,Cepheids-SNIa,Feeney,Y,2017,SNIa,73.2,1.8,1.8,1707.00007
Direct,Cepheids-SNIa,Riess,Y,2016,R16,73.2,1.7,1.7,1604.01424
Direct,Cepheids-SNIa,Cardona,Y,2016,HPs,73.8,2.1,2.1,1611.06088
Direct,Cepheids-SNIa,Freedman,Y,2012,SNIa,74.3,2.1,2.1,2012ApJ
Direct,TRGB-SNIa,Soltis,Y,2020,SNIa,72.1,2,2,2012.09196
Direct,TRGB-SNIa,Freedman,Y,2020,SNIa,69.6,1.9,1.9,2002.0155
Direct,TRGB-SNIa,Reid,Y,2019,SH0ES,71.1,1.9,1.9,1908.05625
Direct,TRGB-SNIa,Freedman,Y,2019,SNIa,69.8,1.9,1.9,1907.05922
Direct,TRGB-SNIa,Yuan,Y,2019,SNIa,72.4,2,2,1908.00993
Direct,TRGB-SNIa,Jang,Y,2017,SNIa,71.2,2.5,2.5,1702.01118
Direct,GW related,Gayathri,Y,2020,GW170817,73.4,10.7,6.9,2009.14247
Direct,GW related,Mukherjee,Y,2020,GW170817,67.6,4.2,4.3,2009.14199
Direct,GW related,Mukherjee,Y,2019,GW170817,68.3,4.5,4.6,1909.08627
Direct,GW related,Abbott,Y,2017,GW170817,70,8,12,1710.05835
Indirect,CMB with Planck,Aghanim,Y,2020,Planck2020,67.4,0.5,0.5,1807.06209
Indirect,BAO,Eisenstein,Y,2005,BAO (SDSS),70.4,1.5,1.5,0501171
Indirect,BAO,BOSS Collaboration,N,2016,BAO (BOSS),67.6,1.2,1.2,1607.03155
Indirect,CMB without Planck,ACT Collaboration,N,2025,CMB (ACT),67.7,0.4,0.4,N/A
Indirect,BAO,DESI Collaboration,N,2025,DESI+CMB (omegaCDM),69.51,0.92,0.92,2503.14738
Direct,TRGB,Freedman et al,2021,Gaia EDR3 + HST,73.04,71.98,74.10,2502.03705v1
Indirect,Cepheid + SN Ia,Riess,Y,2021,Pantheon+ + SH0ES,73.04,71.27,74.81,2502.03705v1
Direct,TRGB + Cepheid,Freedman,Y,2023,JWST + HST,72.5,71.0,74.0,2502.03705v1
Indirect,Early Dark Energy,Riess,Y,2023,CMB + BAO,73.0,71.5,74.5,2502.03705v1
Direct,Cepheid,This work,N,2023,Konkoly ECD,70.26,7.1,7.1,N/A
Direct,SNIa,This work,N,2024,SOFI-ESO+ZTF+ATLAS,69.49,3.12,3.12,N/A
"""

# Read the updated data into a DataFrame
df = pd.read_csv(StringIO(data))

# Convert relevant columns to numeric
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df['Lower'] = pd.to_numeric(df['Lower'], errors='coerce')
df['Upper'] = pd.to_numeric(df['Upper'], errors='coerce')

# Check for NaN values
print(df.isna().sum())

# Define color mapping for different types
color_dict = {
    'CMB with Planck': 'blue',
    'CMB without Planck': 'orange',
    'Cepheids-SNIa': 'green',
    'TRGB-SNIa': 'red',
    'GW related': 'purple'
}

# Initialize the emissions tracker
tracker = EmissionsTracker()
tracker.start()

# Create the plot
plt.figure(figsize=(12, 8))

# Loop through each type and plot with error bars
for t in df['Type'].unique():
    subset = df[df['Type'] == t]

    # Sort by Year to ensure proper plotting
    subset = subset.sort_values('Year')

    # Create an offset for y-values to avoid overlap
    y_offset = 0.1 * subset.index  # Adjust the multiplier as needed for spacing

    # Plot the fill between for the range of values
    plt.fill_between(
        subset['Year'],
        subset['Value'] - subset['Lower'],
        subset['Value'] + subset['Upper'],
        color=color_dict.get(t, 'black'),
        alpha=0.5,
        label=t if t not in plt.gca().get_legend_handles_labels()[1] else ""  # Avoid duplicate labels
    )

    # Plot the error bars
    plt.errorbar(
        subset['Year'],
        subset['Value'] + y_offset,  # Apply the offset
        yerr=subset[['Lower', 'Upper']].mean(axis=1),  # Average of lower and upper for error bars
        fmt='o',
        color=color_dict.get(t, 'black'),  # Default to black if type not in dictionary
        capsize=5
    )

# Adding labels and title
plt.title('Hubble Constant Measurements Over Time')
plt.xlabel('Year')
plt.ylabel('Hubble Constant (H0)')
plt.axhline(y=70, color='red', linestyle='--', label='H0 = 70 km/s/Mpc')
plt.legend()
plt.grid()

# Show plot
plt.show()

# Stop the emissions tracker
tracker.stop()