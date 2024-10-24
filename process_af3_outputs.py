import os
import json
import argparse
import pandas as pd
import numpy as np
from Bio.PDB import NeighborSearch
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import MMCIFIO, Select, Structure, Chain, Model
from Bio.PDB.Superimposer import Superimposer
import logging
from time import sleep
import csv
import subprocess

# Configure logging to include more detailed information
logging.basicConfig(
    filename='process_af3_outputs.log',  # Log file to capture detailed output
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_dot_files(directory):
    """
    Cleans hidden macOS dot files (e.g., .DS_Store) from the specified directory.

    Parameters:
        directory (str): The path to the directory to clean.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    try:
        result = subprocess.run(['dot_clean', directory], capture_output=True, text=True, check=True)
        print(f"dot_clean output: {result.stdout}")
        logging.info(f"Successfully cleaned dot files in the directory: {directory}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running dot_clean: {e.stderr}")

def read_cif_file(file_path, retries=3):
    """
    Attempts to read a CIF file with robust error handling and logging.
    
    Parameters:
        file_path (str): The path to the CIF file.
        retries (int): The number of retry attempts if reading fails.
    
    Returns:
        str: The content of the CIF file if successfully read, otherwise None.
    """
    attempt = 0
    encodings = ['utf-8', 'iso-8859-1']
    while attempt < retries:
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    data = f.read()
                logging.info(f"Successfully read CIF file: {file_path} with {encoding} encoding.")
                return data
            except UnicodeDecodeError as e:
                logging.warning(f"UnicodeDecodeError on {file_path} with {encoding} encoding: {e}.")
            except Exception as e:
                logging.error(f"Error reading CIF file {file_path} on attempt {attempt + 1}: {e}")
        attempt += 1
        sleep(1)  # Small delay before retrying

    logging.critical(f"Failed to read CIF file after {retries} attempts: {file_path}")
    return None

def check_interaction_criteria(summary_file, poi_chain, partner_chain, max_pae_cutoff, min_iptm_cutoff, min_ptm_cutoff):
    """
    Checks if the interaction criteria are met based on the summary file.

    Parameters:
        summary_file (str): The path to the summary file.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        max_pae_cutoff (float): The maximum PAE cutoff value.
        min_iptm_cutoff (float): The minimum iPTM cutoff value.
        min_ptm_cutoff (float): The minimum PTM cutoff value.

    Returns:
        bool: True if the interaction criteria are met, False otherwise.
    """
    try:
        with open(summary_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except UnicodeDecodeError:
        try:
            with open(summary_file, 'r', encoding='latin-1') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            return False  # Error, return False indicating not a binder
    except json.JSONDecodeError:
        return False  # Error, return False indicating not a binder
    except Exception as e:
        logging.error(f"Unexpected error reading {summary_file}: {e}")
        return False  # Error, return False indicating not a binder

    iptm = data.get("iptm", 0)
    ptm = data.get("ptm", 0)
    chain_pair_pae_min = data.get("chain_pair_pae_min", [])

    if iptm < min_iptm_cutoff or ptm < min_ptm_cutoff:
        return False

    chain_indices = {chain: idx for idx, chain in enumerate(['A', 'B', 'C', 'D', 'E'])}
    poi_idx = chain_indices.get(poi_chain)
    partner_idx = chain_indices.get(partner_chain)

    if poi_idx is None or partner_idx is None:
        return False

    try:
        pae_value = chain_pair_pae_min[poi_idx][partner_idx]
    except IndexError:
        logging.error(f"Index error accessing PAE values for {poi_chain} to {partner_chain}")
        return False

    if pae_value >= max_pae_cutoff:
        return False

    return True  # No errors and all checks passed

def extract_pae_data(json_file_path):
    """
    Extracts PAE data from a JSON file and saves it as a CSV file.

    Parameters:
        json_file_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: The PAE data as a DataFrame if successfully extracted, otherwise None.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        pae_data = data.get('pae')
        token_res_ids = data.get('token_res_ids')
        if pae_data is not None and token_res_ids is not None:
            pae_df = pd.DataFrame(pae_data)
            output_file_path = os.path.splitext(json_file_path)[0] + '_pae.csv'
            pae_df.to_csv(output_file_path, index=False)
            logging.info(f"PAE data saved to {output_file_path}")
            return pae_df
        else:
            logging.error("JSON file does not contain 'pae' or 'token_res_ids' keys.")
            return None
    except Exception as e:
        logging.error(f"Error reading the JSON file: {e}")
        return None

def extract_chain_info(cif_file):
    """
    Extracts chain information from a CIF file, handling potential encoding issues.

    Parameters:
    cif_file (str): Path to the CIF file.

    Returns:
    dict: Dictionary with chain information.
    """
    parser = MMCIFParser(QUIET=True)
    chain_info = {}

    # Read CIF file with robust error handling
    data = read_cif_file(cif_file)
    if not data:
        return chain_info

    # Attempt to parse structure from CIF data
    try:
        structure = parser.get_structure("model_0", cif_file)
        logging.info(f"Successfully parsed structure from CIF file: {cif_file}")
    except Exception as e:
        logging.error(f"Failed to parse structure from CIF file: {cif_file} - {e}")
        return chain_info

    # Process the structure to extract chain information
    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())
            sequence = ''.join([seq1_dict.get(residue.resname, 'X') for residue in residues])
            residue_info = []
            residue_length = 0

            for residue in residues:
                residue_id = residue.id[1]
                residue_name = residue.resname
                atom_count = len(residue.child_list)
                residue_info.append((residue_id, residue_name, atom_count))
                if residue_name in seq1_dict:
                    residue_length += 1
                else:
                    residue_length += atom_count

            chain_info[chain.id] = {
                'residue_length': residue_length,
                'sequence': sequence,
                'residues': residue_info
            }

    return chain_info

seq1_dict = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

def identify_interacting_residues(pae_df, chain_lengths, poi_chain, partner_chain, max_pae_cutoff, min_residues):
    """
    Identifies interacting residues based on PAE data and chain lengths.

    Parameters:
        pae_df (pd.DataFrame): The PAE data as a DataFrame.
        chain_lengths (list): List of chain lengths.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        max_pae_cutoff (float): The maximum PAE cutoff value.
        min_residues (int): The minimum number of residues below the PAE cutoff.

    Returns:
        list: List of interacting residues.
    """
    chain_indices = {chain: idx for idx, chain in enumerate(['A', 'B', 'C', 'D', 'E'])}
    poi_idx = chain_indices.get(poi_chain)
    partner_idx = chain_indices.get(partner_chain)

    if poi_idx is None or partner_idx is None:
        logging.error(f"Invalid chain identifier: {poi_chain} or {partner_chain}")
        return []

    logging.info(f"Chain Indices - POI: {poi_idx}, Partner: {partner_idx}")
    logging.info(f"Chain Lengths: {chain_lengths}")

    if poi_idx >= len(chain_lengths) or partner_idx >= len(chain_lengths):
        logging.error(f"Chain index out of bounds: POI index {poi_idx}, Partner index {partner_idx}, Chain lengths {len(chain_lengths)}")
        return []

    start_poi = sum(chain_lengths[:poi_idx])
    end_poi = start_poi + chain_lengths[poi_idx]
    start_partner = sum(chain_lengths[:partner_idx])
    end_partner = start_partner + chain_lengths[partner_idx]

    logging.info(f"PAE DataFrame shape: {pae_df.shape}")
    logging.info(f"Start POI: {start_poi}, End POI: {end_poi}, Start Partner: {start_partner}, End Partner: {end_partner}")

    if start_partner >= pae_df.shape[1] or end_partner > pae_df.shape[1]:
        logging.error(f"Index out of bounds: start_partner {start_partner}, end_partner {end_partner}, DataFrame width {pae_df.shape[1]}")
        return []

    interacting_residues = []

    pae_df_t = pae_df.T

    residues_meeting_criteria = pae_df_t.iloc[start_partner:end_partner].apply(
        lambda col: sum(col[start_poi:end_poi] < max_pae_cutoff) >= min_residues,
        axis=1
    )

    interacting_residues = [int(x) - start_partner + 1 for x in residues_meeting_criteria.index[residues_meeting_criteria]]
    
    return interacting_residues

def find_contact_residues(cif_file, poi_chain, partner_chain, interacting_residues, max_dist):
    """
    Finds contact residues between the protein of interest and the partner protein.

    Parameters:
        cif_file (str): The path to the CIF file.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        interacting_residues (list): List of interacting residues.
        max_dist (float): The maximum distance for contact residues.

    Returns:
        dict: Dictionary mapping partner residues to contact residues in the protein of interest.
    """
    parser = MMCIFParser()
    structure = parser.get_structure("model_0", cif_file)

    poi_atoms = [atom for residue in structure[0][poi_chain] for atom in residue if is_aa(residue)]
    partner_atoms = [atom for residue in structure[0][partner_chain] for atom in residue if is_aa(residue) and residue.id[1] in interacting_residues]

    neighbor_search = NeighborSearch(poi_atoms + partner_atoms)
    contact_map = {}

    for residue in structure[0][partner_chain]:
        if is_aa(residue) and residue.id[1] in interacting_residues:
            partner_residue_id = residue.id[1]
            contact_residues = []

            for atom in residue:
                nearby_atoms = neighbor_search.search(atom.coord, max_dist)
                for nearby_atom in nearby_atoms:
                    if nearby_atom.get_parent().get_parent().id == poi_chain:
                        poi_residue = nearby_atom.get_parent()
                        contact_residues.append(poi_residue.id[1])

            if contact_residues:
                contact_map[partner_residue_id] = list(set(contact_residues))

    return contact_map

def find_consecutive_groups(numbers, max_gap=2, min_length=3):
    """
    Find consecutive groups of numbers in a list, allowing for a specified maximum gap
    between numbers to still consider them consecutive.
    
    Parameters:
        numbers (list): A sorted list of numbers to group.
        max_gap (int): The maximum gap between numbers to consider them consecutive.
        min_length (int): The minimum length of a group to be considered valid.
    
    Returns:
        list: A list of lists, where each sublist is a group of consecutive numbers.
    """
    if not numbers:
        logging.warning("The input list 'numbers' is empty.")
        return []

    logging.info(f"Processing numbers: {numbers}")

    groups = []
    group = [numbers[0]]

    for current, next_ in zip(numbers[:-1], numbers[1:]):
        if next_ - current <= max_gap:
            group.append(next_)
        else:
            if len(group) >= min_length:
                groups.append(group)
            group = [next_]
    if len(group) >= min_length:
        groups.append(group)

    logging.info(f"Found consecutive groups: {groups}")
    return groups

def process_consecutive_interactions(contact_map):
    """
    Processes consecutive interactions from the contact map.

    Parameters:
        contact_map (dict): Dictionary mapping partner residues to contact residues in the protein of interest.

    Returns:
        dict: Dictionary mapping consecutive partner residue groups to consecutive contact residue groups.
    """
    consecutive_interactions = {}

    sorted_keys = sorted(contact_map.keys())
    consecutive_keys_groups = find_consecutive_groups(sorted_keys, max_gap=1, min_length=3)

    for group in consecutive_keys_groups:
        contact_residues = set()
        for key in group:
            contact_residues.update(contact_map[key])
        
        consecutive_contact_groups = find_consecutive_groups(list(contact_residues), max_gap=2, min_length=3)
        consecutive_interactions[tuple(group)] = consecutive_contact_groups

    return consecutive_interactions

def save_to_csv(data, poi_chain, partner_chain, max_pae_cutoff, max_dist):
    """
    Saves the interaction analysis data to a CSV file.

    Parameters:
        data (list): List of dictionaries containing interaction analysis data.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        max_pae_cutoff (float): The maximum PAE cutoff value.
        max_dist (float): The maximum distance for contact residues.
    """
    filename = f'interaction_analysis_PAE_{max_pae_cutoff}_max_dist_{max_dist}.csv'
    fieldnames = [
        'Folder_name', 
        f'Contact_residues_POI_chain_{poi_chain}', 
        'Contact_sequence', 
        f'Interacting_residues_Partner_chain_{partner_chain}', 
        'Interacting_sequence'
    ]
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

class ResidueSelect(Select):
    def __init__(self, poi_chain, partner_chain, consecutive_interactions):
        self.poi_chain = poi_chain
        self.partner_chain = partner_chain
        self.consecutive_interactions = consecutive_interactions

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        res_id = residue.id[1]

        if chain_id == self.poi_chain:
            return True
        if chain_id == self.partner_chain:
            for group in self.consecutive_interactions:
                if res_id in group:
                    return True
        return False

def create_interaction_cif(original_cif_file, output_cif_file, poi_chain, partner_chain, consecutive_interactions):
    """
    Creates a CIF file containing only the interacting residues.

    Parameters:
        original_cif_file (str): The path to the original CIF file.
        output_cif_file (str): The path to the output CIF file.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        consecutive_interactions (list): List of consecutive interaction groups.
    """
    parser = MMCIFParser()
    structure = parser.get_structure("model_0", original_cif_file)
    io = MMCIFIO()
    io.set_structure(structure)
    
    select = ResidueSelect(poi_chain, partner_chain, consecutive_interactions)
    io.save(output_cif_file, select=select)

def process_full_data_files(binder_dir, poi_chain, partner_chain, max_pae_cutoff, min_residues, max_dist, collected_data, output_dir):
    """
    Processes full data files in the binder directory and extracts interaction information.

    Parameters:
        binder_dir (str): The path to the binder directory.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        max_pae_cutoff (float): The maximum PAE cutoff value.
        min_residues (int): The minimum number of residues below the PAE cutoff.
        max_dist (float): The maximum distance for contact residues.
        collected_data (list): List to collect interaction analysis data.
        output_dir (str): The path to the output directory for saving interaction CIF files.
    """
    dir_name = os.path.basename(binder_dir)
    cif_file = os.path.join(binder_dir, f"{dir_name}_model_0.cif")

    if os.path.exists(cif_file):
        chain_info = extract_chain_info(cif_file)
        chain_lengths = [info['residue_length'] for info in chain_info.values()]
        if not chain_lengths:
            return

        json_file_path = os.path.join(binder_dir, f"{dir_name}_full_data_0.json")
        if os.path.exists(json_file_path):
            pae_df = extract_pae_data(json_file_path)
            if pae_df is not None:
                interacting_residues = identify_interacting_residues(pae_df, chain_lengths, poi_chain, partner_chain, max_pae_cutoff, min_residues)
                contact_map = find_contact_residues(cif_file, poi_chain, partner_chain, interacting_residues, max_dist)
                consecutive_interactions = process_consecutive_interactions(contact_map)
                
                logging.info(f"Interacting residues in {binder_dir}: {interacting_residues}")
                logging.info(f"Contact residues in {binder_dir}: {contact_map}")
                logging.info(f"Consecutive interactions in {binder_dir}: {consecutive_interactions}")

                poi_sequence = chain_info[poi_chain]['sequence']
                partner_sequence = chain_info[partner_chain]['sequence']

                for interacting_group, contact_groups in consecutive_interactions.items():
                    for contact_group in contact_groups:
                        collected_data.append({
                            'Folder_name': dir_name,
                            f'Contact_residues_POI_chain_{poi_chain}': f"{min(contact_group)}-{max(contact_group)}",
                            'Contact_sequence': poi_sequence[min(contact_group)-1:max(contact_group)],
                            f'Interacting_residues_Partner_chain_{partner_chain}': f"{min(interacting_group)}-{max(interacting_group)}",
                            'Interacting_sequence': partner_sequence[min(interacting_group)-1:max(interacting_group)]
                        })
                
                output_cif_file = os.path.join(output_dir, f"{dir_name}_interaction.cif")
                create_interaction_cif(cif_file, output_cif_file, poi_chain, partner_chain, list(consecutive_interactions.keys()))
        else:
            logging.error(f"JSON file not found in directory: {json_file_path}")
    else:
        logging.error(f"CIF file not found in directory: {cif_file}")

def extract_and_save_model(cif_file, poi_chain, partner_chain, consecutive_interactions, model_index, output_dir):
    """
    Extracts the POI and relevant partner protein regions from a CIF file,
    and saves them into a new CIF file named as model_X.cif (e.g., model_0.cif).

    Parameters:
        cif_file (str): The path to the CIF file.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        consecutive_interactions (list): List of consecutive interaction groups.
        model_index (int): The model index.
        output_dir (str): The path to the output directory for saving the extracted model.
    """
    logging.info(f"Extracting and saving model from {cif_file} (model index: {model_index})")
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(f"model_{model_index}", cif_file)
        logging.info(f"Successfully parsed structure from {cif_file}")

        # Create new chains with consistent labels: 'A' for POI, 'B' for partner
        poi_chain_to_add = Chain.Chain('A')
        partner_chain_to_add = Chain.Chain('B')

        # Extract POI chain
        for residue in structure[0][poi_chain]:
            poi_chain_to_add.add(residue.copy())

        # Extract relevant partner residues
        for residue in structure[0][partner_chain]:
            if any(residue.id[1] in group for group in consecutive_interactions):
                partner_chain_to_add.add(residue.copy())

        # Create a new model named model_X
        model = Model.Model(f"model_{model_index}")
        model.add(poi_chain_to_add)
        model.add(partner_chain_to_add)

        # Create a new structure for saving
        new_structure = Structure.Structure(f"model_{model_index}")
        new_structure.add(model)

        # Save the structure to a new CIF file named model_X.cif
        output_cif_file = os.path.join(output_dir, f"model_{model_index}.cif")
        io = MMCIFIO()
        io.set_structure(new_structure)
        io.save(output_cif_file)
        logging.info(f"Saved CIF file: {output_cif_file}")

    except Exception as e:
        logging.error(f"Failed to extract and save model from {cif_file}: {e}")

def process_overlay_files(binder_dir, poi_chain, partner_chain, max_pae_cutoff, min_residues, max_dist, overlay_output_dir):
    """
    Processes all CIF files in a directory, creating individual CIF files
    for each model and saving them in a corresponding folder within the overlay directory.
    Generates a PyMOL script (.pml) to align and save the models.

    Parameters:
        binder_dir (str): The path to the binder directory.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        max_pae_cutoff (float): The maximum PAE cutoff value.
        min_residues (int): The minimum number of residues below the PAE cutoff.
        max_dist (float): The maximum distance for contact residues.
        overlay_output_dir (str): The path to the overlay output directory.
    """
    logging.info(f"Processing overlay files in directory: {binder_dir}")
    try:
        cif_files = [os.path.join(binder_dir, f) for f in os.listdir(binder_dir) if f.endswith('.cif')]
        cif_files.sort()  # Ensure correct order of models
        logging.info(f"Found {len(cif_files)} CIF files to process")

        # Create a corresponding subfolder in the overlay directory
        overlay_subfolder = os.path.join(overlay_output_dir, os.path.basename(binder_dir))
        os.makedirs(overlay_subfolder, exist_ok=True)
        logging.info(f"Overlay subfolder created: {overlay_subfolder}")

        # Assume the first structure has the correct interaction criteria information
        chain_info = extract_chain_info(cif_files[0])
        chain_lengths = [info['residue_length'] for info in chain_info.values()]
        if not chain_lengths:
            logging.warning("No chain lengths found; exiting.")
            return

        json_file_path = os.path.join(binder_dir, f"{os.path.basename(binder_dir)}_full_data_0.json")
        if os.path.exists(json_file_path):
            logging.info(f"JSON file found: {json_file_path}")
            pae_df = extract_pae_data(json_file_path)
            if pae_df is not None:
                interacting_residues = identify_interacting_residues(
                    pae_df, chain_lengths, poi_chain, partner_chain, max_pae_cutoff, min_residues)
                contact_map = find_contact_residues(
                    cif_files[0], poi_chain, partner_chain, interacting_residues, max_dist)
                consecutive_interactions = process_consecutive_interactions(contact_map)

                # Process each CIF file and save as individual model_X.cif
                for index, cif_file in enumerate(cif_files):
                    extract_and_save_model(cif_file, poi_chain, partner_chain, consecutive_interactions, index, overlay_subfolder)

                # Create a PyMOL script for alignment and saving
                create_pymol_script(overlay_subfolder, overlay_output_dir)

    except Exception as e:
        logging.error(f"An error occurred while processing overlay files: {e}")

def create_pymol_script(input_folder, output_folder):
    """
    Creates a PyMOL script (.pml) to load, align based on chain A, color by chain, and save models as a .pse file.
    Ensures compatibility with external drives by using absolute paths and checking permissions.

    Parameters:
    input_folder (str): The path to the folder containing CIF files.
    output_folder (str): The path to the output folder for saving the .pse file.

    Returns:
    None
    """
    logging.info(f"Creating PyMOL script for folder: {input_folder}")
    try:
        # Ensure input and output paths are absolute to avoid path resolution issues
        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder)

        # Check if input folder exists and is accessible
        if not os.path.isdir(input_folder):
            logging.error(f"Input folder {input_folder} does not exist or is not accessible.")
            return

        # Attempt to create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Get all CIF files sorted in the correct order
        cif_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.cif')])
        logging.info(f"Found {len(cif_files)} CIF files for PyMOL script")

        # Create the .pml script content
        script_content = ""
        for i, cif_file in enumerate(cif_files):
            model_name = f"model_{i}"
            cif_file_path = os.path.abspath(os.path.join(input_folder, cif_file))
            script_content += f"load {cif_file_path}, {model_name}\n"
        
        # Align only chain A of each model to chain A of model_0
        for i in range(1, len(cif_files)):
            script_content += f"align model_{i} and chain A, model_0 and chain A\n"
        
        # Color each chain differently
        script_content += "util.cbc()\n"

        # Define the output folder name and file path
        folder_name = os.path.basename(input_folder)
        output_pse = os.path.abspath(os.path.join(output_folder, f"{folder_name}_overlay.pse"))

        # Ensure output directory exists before running the script
        output_dir = os.path.dirname(output_pse)
        os.makedirs(output_dir, exist_ok=True)

        # Add command to save the session
        script_content += f"save {output_pse}\n"

        # Save the script to a .pml file in the input folder
        script_path = os.path.join(input_folder, "align_and_save.pml")
        with open(script_path, "w") as script_file:
            script_file.write(script_content)
        logging.info(f"PyMOL script saved at {script_path}")

    except PermissionError as e:
        logging.error(f"Permission error: {e}. Check your access rights to {input_folder} and {output_folder}.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

def process_directory(input_dir, poi_chain, partner_chain, max_pae_cutoff, min_iptm_cutoff, min_ptm_cutoff, min_residues, max_dist):
    """
    Processes the input directory, creating overlay directories and saving individual
    CIF models for each folder, and generates PyMOL scripts for alignment.

    Parameters:
        input_dir (str): The path to the input directory containing subfolders with AlphaFold3 outputs.
        poi_chain (str): The chain identifier for the protein of interest.
        partner_chain (str): The chain identifier for the potential interaction partner.
        max_pae_cutoff (float): The maximum PAE cutoff value.
        min_iptm_cutoff (float): The minimum iPTM cutoff value.
        min_ptm_cutoff (float): The minimum PTM cutoff value.
        min_residues (int): The minimum number of residues below the PAE cutoff.
        max_dist (float): The maximum distance for contact residues.

    Returns:
        dict: Dictionary mapping directory paths to boolean values indicating if they meet the interaction criteria.
    """
    results = {}
    collected_data = []

    # Run dot_clean on the input directory to handle ._ files
    clean_dot_files(input_dir)

    # Create output directories for saving interaction and overlay files
    interaction_output_dir = f"Interaction_cif_files_PAE_{max_pae_cutoff}_maxdist_{max_dist}"
    overlay_output_dir = f"Overlays_Interaction_cif_files_PAE_{max_pae_cutoff}_maxdist_{max_dist}"
    os.makedirs(interaction_output_dir, exist_ok=True)
    os.makedirs(overlay_output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Skip hidden files and other non-standard files
            if file.startswith("._") or not file.endswith("_summary_confidences_0.json"):
                continue

            summary_file = os.path.join(root, file)
            is_binder = check_interaction_criteria(summary_file, poi_chain, partner_chain, max_pae_cutoff, min_iptm_cutoff, min_ptm_cutoff)
            results[root] = is_binder
            if is_binder:
                # Process interaction CIF files
                process_full_data_files(root, poi_chain, partner_chain, max_pae_cutoff, min_residues, max_dist, collected_data, interaction_output_dir)
                
                # Process overlay CIF files and generate PyMOL script
                process_overlay_files(root, poi_chain, partner_chain, max_pae_cutoff, min_residues, max_dist, overlay_output_dir)

            logging.info(f"Directory: {root}, Binder: {is_binder}")

    save_to_csv(collected_data, poi_chain, partner_chain, max_pae_cutoff, max_dist)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AlphaFold3 outputs for protein-protein interaction screens.")
    parser.add_argument("-id", "--input_dir", required=True, help="Input directory containing subfolders with AlphaFold3 outputs.")
    parser.add_argument("-poi", "--poi_chain", default="A", help="Chain on which the Protein of interest (POI) is located (default: A).")
    parser.add_argument("-partner", "--partner_chain", default="B", help="Chain on which potentially predicted interaction partner is located (default: B).")
    parser.add_argument("-pae", "--max_pae_cutoff", type=float, default=15.0, help="Maximum PAE cutoff (default: 15).")
    parser.add_argument("-iptm", "--min_iptm_cutoff", type=float, default=0.0, help="Minimum iPTM cutoff (default: 0).")
    parser.add_argument("-ptm", "--min_ptm_cutoff", type=float, default=0.0, help="Minimum PTM cutoff (default: 0).")
    parser.add_argument("-min_residues", "--min_residues_cutoff", type=int, default=5, help="Minimum number of residues below the PAE cutoff (default: 5).")
    parser.add_argument("-max_dist", "--max_dist", type=float, default=8.0, help="Maximum distance for contact residues (default: 8.0 Å).")

    args = parser.parse_args()

    results = process_directory(args.input_dir, args.poi_chain, args.partner_chain, args.max_pae_cutoff, args.min_iptm_cutoff, args.min_ptm_cutoff, args.min_residues_cutoff, args.max_dist)
    
    logging.info("\nSummary of results:")
    for dir, is_binder in results.items():
        logging.info(f"Directory: {dir}, Binder: {is_binder}")
