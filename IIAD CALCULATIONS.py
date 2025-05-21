import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_data(file_name, width=25e-3, thickness=5e-3, gauge_length=144e-3):
    """Process raw data from excel file and return processed dataframe"""
    cross_sectional_area = width * thickness
    
    data = pd.read_excel(file_name)
    # Subtract initial load value from all load measurements
    initial_load = data["Load N"].iloc[0]
    data["Load N"] = data["Load N"] - initial_load
    
    if "test5" in file_name:
        data["Deformation m"] = data["Deformation 1 microm"] * 1e-3

    else:
        data["Deformation m"] = data["Deformation 1 microm"] * 1e-6
        
    data["Strain"] = data["Deformation m"] / gauge_length
    data["Stress (Pa)"] = data["Load N"] / cross_sectional_area
    
    # Trim data to 80% of max stress
    max_stress = data["Stress (Pa)"].max()
    threshold = 0.8 * max_stress
    max_valid_idx = data[data["Stress (Pa)"] >= threshold].index[-1]
    data = data.iloc[:max_valid_idx + 1]
    
    return data

def find_linear_region(strain, stress, window_size=200, filename=None):
    """Find the most linear region using sliding window and R² criterion"""
    max_r2 = 0
    best_start = 0
    
    # Use 20% cutoff for test5, 50% for others
    if filename and "test5" in filename:
        max_idx = int(len(strain) * 0.1)
    else:
        max_idx = int(len(strain) * 0.5)
    
    for start in range(0, max_idx - window_size):
        end = start + window_size
        strain_window = strain[start:end]
        stress_window = stress[start:end]
        
        try:
            # Reshape data for polyfit
            X = strain_window.values.reshape(-1, 1)
            y = stress_window.values
            
            # Calculate R² score
            coeffs = np.polyfit(strain_window, stress_window, 1)
            y_pred = np.polyval(coeffs, strain_window)
            r2 = 1 - np.sum((stress_window - y_pred) ** 2) / np.sum((stress_window - np.mean(stress_window)) ** 2)
            
            if r2 > max_r2:
                max_r2 = r2
                best_start = start
                
        except np.linalg.LinAlgError:
            continue
            
    return best_start, best_start + window_size

def calculate_yield_strength(data, offset_strain=0.001, filename=None):
    """Calculate yield strength using 0.1% offset method"""
    # Find most linear region
    start_idx, end_idx = find_linear_region(data["Strain"], data["Stress (Pa)"], filename=filename)
    
    # Use identified linear region
    strain_elastic = data["Strain"].iloc[start_idx:end_idx]
    stress_elastic = data["Stress (Pa)"].iloc[start_idx:end_idx]
    
    elastic_modulus = np.polyfit(strain_elastic, stress_elastic, 1)[0]
    offset_line = elastic_modulus * (data["Strain"] - offset_strain)
    
    difference = abs(data["Stress (Pa)"] - offset_line)
    idx_yield = difference.idxmin()
    
    return data["Stress (Pa)"].iloc[idx_yield], idx_yield, offset_line

def calculate_uts(data):
    """Calculate Ultimate Tensile Stress"""
    return data["Stress (Pa)"].max()

def calculate_total_strain(data):
    """Calculate total strain at failure"""
    return data["Strain"].iloc[-1]

def calculate_fracture_energy(data):
    """Calculate fracture energy using trapezoidal integration"""
    return np.trapz(data["Stress (Pa)"], data["Strain"])

def plot_single_curve(data, yield_strength, idx_yield, offset_line, uts, ax=None):
    """Plot stress-strain curve with key points for a single dataset"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data["Strain"], data["Stress (Pa)"], label="Stress-Strain Curve")
    ax.fill_between(data["Strain"], data["Stress (Pa)"], alpha=0.3)
    #ax.plot(data["Strain"], offset_line, "--", label="0.1% Offset Line", color='gray')
    ax.scatter(data["Strain"].iloc[idx_yield], yield_strength, color="red", label="Yield Point")
    ax.scatter(data["Strain"][data["Stress (Pa)"].idxmax()], uts, 
              color="green", label="Ultimate Tensile Stress")
    
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (Pa)")
    ax.grid(True)
    
    return ax

def analyze_multiple_files(file_names):
    """Analyze multiple files and create plots"""
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))
    
    results = []
    
    for i, file_name in enumerate(file_names):
        data = process_data(file_name)
        yield_strength, idx_yield, offset_line = calculate_yield_strength(data, filename=file_name)
        uts = calculate_uts(data)
        total_strain = calculate_total_strain(data)
        fracture_energy = calculate_fracture_energy(data)
        
        # Store results
        results.append({
            'file': file_name,
            'yield_strength': yield_strength,
            'uts': uts,
            'total_strain': total_strain,
            'fracture_energy': fracture_energy
        })
        
        # Individual plot in separate window
        fig_individual = plt.figure(figsize=(10, 6))
        ax = fig_individual.add_subplot(111)
        ax = plot_single_curve(data, yield_strength, idx_yield, offset_line, uts, ax=ax)
        ax.set_title(f"Sample {i+1}")
        ax.legend()
        
        # Add to combined plot
        ax_combined.plot(data["Strain"], data["Stress (Pa)"], label=f"Sample {i+1}")
    
    ax_combined.set_xlabel("Strain")
    ax_combined.set_ylabel("Stress (Pa)")
    ax_combined.set_title("Combined Stress-Strain Curves")
    ax_combined.legend()
    ax_combined.grid(True)
    
    plt.show()
    
    return results


# write here the path to the tests folder on your own computer!!
TEST_DIR = "tests"

file_names = [
    f"{TEST_DIR}/test1.xlsx", # test 1_1 , specimen from Rafael
    f"{TEST_DIR}/test2.xlsx", # test 1_2 , specimen from Wao
    f"{TEST_DIR}/test3.xlsx", # test 2_1 , specimen from Ian
    f"{TEST_DIR}/test4.xlsx", # test 2_2 , specimen from Ian
    f"{TEST_DIR}/test5.xlsx", # final test, specimen from Rafael again
    
    # (add files in chronological order)
]

results = analyze_multiple_files(file_names)


for result in results:
    print(f"\nResults for {result['file']}:")
    print(f"Yield Strength: {result['yield_strength']:.2f} Pa")
    print(f"Ultimate Tensile Stress: {result['uts']:.2f} Pa")
    print(f"Total Strain at Failure: {result['total_strain']:.4f}")
    print(f"Fracture Energy: {result['fracture_energy']:.2f} J/m³")
