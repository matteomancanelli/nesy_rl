import xml.etree.ElementTree as ET
import ast
import pathlib
import torch
import numpy as np
import os

from trace_alignment.trace_alignment_handler import align_traces

def expand_dataset_with_end_of_trace_symbol(original_dataset):
    """
    Expand the original dataset by appending an "end_of_sequence" symbol to each sequence.

    Args:
    original_dataset (torch.Tensor): Original dataset with shape (batch_size, len_sequences, num_symbols).

    Returns:
    torch.Tensor: Expanded dataset with shape (batch_size, len_sequences+1, num_symbols+1).
    """
    batch_size, len_sequences, num_symbols = original_dataset.size()

    # Expand dimensions of original dataset to accommodate end_of_sequence symbol
    expanded_dataset = torch.cat((original_dataset, torch.zeros(batch_size, len_sequences, 1)), dim=2)

    # Create end_of_sequence one-hot vector
    end_of_sequence = torch.zeros(batch_size, 1, num_symbols + 1)
    end_of_sequence[:, :, -1] = 1

    # Concatenate end_of_sequence to each sequence
    expanded_dataset = torch.cat((expanded_dataset, end_of_sequence), dim=1)

    return expanded_dataset

# Filters traces in a log file by the specified trace length and writes them to an output file.
def filter_traces_by_length(log_file, out_file, length):
    tree = ET.parse(log_file)
    root = tree.getroot()

    selected_traces = []

    for trace in root.iter('trace'):
        events = trace.findall('event')
        
        if len(events) == length:
            selected_traces.append(trace)

    new_root = ET.Element(root.tag, root.attrib)

    for trace in selected_traces:
        new_root.append(trace)

    new_tree = ET.ElementTree(new_root)
    new_tree.write(out_file, encoding='UTF-8', xml_declaration=True)

    return len(selected_traces)

# Combines multiple log files into a single log file by appending all traces.
def combine_log_files(log_file_lst, out_file):
    tree = ET.parse(log_file_lst[0])
    root = tree.getroot()

    new_root = ET.Element(root.tag, root.attrib)

    for file in log_file_lst:
        tree = ET.parse(file)
        root = tree.getroot()
        
        for trace in root.iter('trace'):
            new_root.append(trace)

    new_tree = ET.ElementTree(new_root)
    new_tree.write(out_file, encoding='UTF-8', xml_declaration=True)

# Filters traces from a dataset folder based on specified trace lengths and combines them if necessary.
def filter_dataset(data_folder, data_file, trace_length_lst):
    filtered_file_name =  "l" + '-'.join(str(l) for l in trace_length_lst)
    log_file_lst = []

    for length in trace_length_lst:
        filtered_file = pathlib.Path(data_folder, f"l{length}.xes")
        filter_traces_by_length(data_file, filtered_file, length)
        log_file_lst.append(filtered_file)
    
    if len(trace_length_lst) > 1:
        filtered_file = pathlib.Path(data_folder, f"{filtered_file_name}.xes")
        combine_log_files(log_file_lst, filtered_file)
    
    return filtered_file_name, filtered_file

# Extracts symbols from a log file and writes the processed traces to a file.
def log_to_traces(log_file, traces_file, max_length):
    tree = ET.parse(log_file)
    root = tree.getroot()

    with open(traces_file, "w") as f:
        for trace in root.iter('trace'):
            symbol_list = []

            for event in trace.iter('event'):
                for string in event.findall("string"):
                    if string.attrib.get('key') == 'lifecycle:transition':
                        transaction = string.attrib.get('value').lower()
                    if string.attrib.get('key') == 'concept:name':
                        concept = string.attrib.get('value').lower().replace(" ", "_")
                
                symbol = concept + "_" + transaction
                symbol_list.append(symbol)

            while len(symbol_list) < max_length + 1:
                symbol_list.append("end")
            
            f.write(f"{symbol_list}\n")

def traces_to_xes(traces_file, log_file):
    # Create the root element with necessary attributes
    root = ET.Element('log')
    root.set('xes.version', '1.0')
    root.set('xes.features', 'nested-attributes')
    root.set('openxes.version', '1.0RC7')
    
    # Read traces from the file
    with open(traces_file, 'r') as f:
        traces = f.readlines()
    
    # Process each trace
    for trace_idx, trace in enumerate(traces):
        # Convert string representation of list to actual list
        events = ast.literal_eval(trace.strip())
        
        # Create trace element
        trace_element = ET.SubElement(root, 'trace')
        
        # Add trace attributes
        trace_id = ET.SubElement(trace_element, 'string')
        trace_id.set('key', 'concept:name')
        trace_id.set('value', str(trace_idx + 1))
        
        # Process each event in the trace
        for event in events:
            if event == "end":
                break
                
            # Create event element
            event_element = ET.SubElement(trace_element, 'event')
            
            # Check if event has concept and transition (separated by "_")
            if "_" in event:
                # Split the event string into concept name and lifecycle transition
                concept, transition = event.rsplit('_', 1)
                
                # Add concept:name
                concept_name = ET.SubElement(event_element, 'string')
                concept_name.set('key', 'concept:name')
                concept_name.set('value', concept.upper())
                
                # Add lifecycle:transition
                lifecycle = ET.SubElement(event_element, 'string')
                lifecycle.set('key', 'lifecycle:transition')
                lifecycle.set('value', transition.upper())
            else:
                # Event has only concept name, no transition information
                concept_name = ET.SubElement(event_element, 'string')
                concept_name.set('key', 'concept:name')
                concept_name.set('value', event.upper())
                
                # Add a default lifecycle:transition if needed
                lifecycle = ET.SubElement(event_element, 'string')
                lifecycle.set('key', 'lifecycle:transition')
                lifecycle.set('value', 'COMPLETE')  # Default value, can be changed
    
    # Create the XML tree and write it to file
    tree = ET.ElementTree(root)
    tree.write(log_file, encoding='UTF-8', xml_declaration=True)

# Converts traces from a file into a Scarlet-readable one-hot encoded format..
def traces_to_scarlet(traces_file, scarlet_file, alphabet):
    with open(traces_file, "r") as input:
        with open(scarlet_file, "w") as output:
            for line in input:
                trace = eval(line.rstrip("\n"))
                one_hot = list_to_one_hot(trace, alphabet)
                output.write(";".join([",".join(map(lambda x: str(x), seq)) for seq in one_hot]) + "\n")

# Converts traces from a file into an STLNet-readable one-hot encoded format.
def traces_to_stlnet(traces_file, stlnet_file, alphabet):
    with open(traces_file, "r") as input:
        with open(stlnet_file, "w") as output:
            for line in input:
                trace = eval(line.rstrip("\n"))
                one_hot = list_to_one_hot(trace, alphabet)
                output.write(" ".join([" ".join(map(lambda x: str(x), seq)) for seq in one_hot]) + "\n")

def stlnet_to_traces(stlnet_file, traces_file, alphabet):
    with open(stlnet_file, "r") as input:
        with open(traces_file, "w") as output:
            for line in input:
                # Split the line into individual numbers
                numbers = line.strip().split()
                
                # Convert string numbers to integers
                one_hot_data = list(map(int, numbers))
                
                # Calculate number of events based on alphabet size
                event_length = len(alphabet)
                num_events = len(one_hot_data) // event_length
                
                # Reconstruct the trace
                trace = []
                for i in range(num_events):
                    # Extract one-hot vector for current event
                    start_idx = i * event_length
                    one_hot_vector = one_hot_data[start_idx:start_idx + event_length]
                    
                    # Convert one-hot vector back to symbol
                    symbol = one_hot_to_symbol(one_hot_vector, alphabet)
                    if symbol is not None:
                        trace.append(symbol)
                
                # Write the trace to output file
                output.write(f"{trace}\n")

def tensor_to_stlnet(tensor, stlnet_file):
    # Flatten the middle and last dimensions while keeping the traces separate
    flattened = tensor.view(tensor.size(0), -1)
    
    # Convert to numpy for easier text processing
    numpy_data = flattened.cpu().numpy()
    
    # Write to file with space-separated format
    with open(stlnet_file, 'w') as f:
        for trace in numpy_data:
            # Convert numbers to strings and join with spaces
            line = ' '.join(map(lambda x: str(int(x)), trace))
            f.write(line + '\n')

def discard_transition(traces_file):
    with open(traces_file, 'r') as file:
        content = file.read()
    
    # Remove '_complete' substring
    modified_content = content.replace('_complete', '')
    
    # Write back to the same file
    with open(traces_file, 'w') as file:
        file.write(modified_content)

def standardize_trace_length(traces_file):
    with open(traces_file, 'r') as f:
        traces = f.readlines()
    
    max_length = 1 + max([len(eval(trace.strip())) for trace in traces])
    standardized_traces = []
    
    for trace_line in traces:
        trace = eval(trace_line.strip())
        
        if len(trace) < max_length:
            padding = ['end'] * (max_length - len(trace))
            trace = trace + padding
        
        standardized_traces.append(trace)
    
    with open(traces_file, 'w') as f:
        for trace in standardized_traces:
            f.write(str(trace) + '\n')
    
    return standardized_traces


# Constructs a mutex formula ensuring mutual exclusivity of symbols in the alphabet.
def get_mutex(alphabet):
    mutex_str = " & ".join(["(" + symbol + " i (" + " & ".join(["! " + sym for sym in alphabet if sym != symbol]) + "))" for symbol in alphabet])
    mutex_str = mutex_str + " & (" + " | ".join([symbol for symbol in alphabet]) + ")"
    return "(G(" + mutex_str + "))"

def list_to_one_hot(trace, alphabet):
    """Converts a list of symbols to one-hot encoded sequences."""
    one_hot = []
    for symbol in trace:
        encoding = [1 if symbol == s else 0 for s in alphabet]
        one_hot.append(encoding)
    return one_hot

def one_hot_to_symbol(one_hot_vector, alphabet):
    """Converts a one-hot encoded vector back to its symbol."""
    try:
        index = one_hot_vector.index(1)
        return alphabet[index]
    except ValueError:
        return None


def write_formula(formula, alphabet):
    for symbol in alphabet:
        formula = formula.replace(symbol, symbol + "_complete")
    
    # Create directories if they don't exist
    form_dir = pathlib.Path("formulas/syntetic")
    form_dir.mkdir(parents=True, exist_ok=True)
    
    form_file = form_dir / "curr_formula.txt"
    with open(form_file, "w") as f:
        f.write(formula)
    
    return form_file


def align_predicted_traces(test_predicted_traces, form_file, alphabet, trace_length):
    data_folder = pathlib.Path("datasets")
    predicted_log_file = pathlib.Path(data_folder, f"predicted.xes")
    predicted_traces_file = pathlib.Path(data_folder, f"predicted.txt")
    predicted_stlnet_file = pathlib.Path(data_folder, f"predicted_stlnet.dat")

    predicted_traces_file0 = pathlib.Path(data_folder, f"predicted0.txt")
    predicted_stlnet_file0 = pathlib.Path(data_folder, f"predicted_stlnet0.dat")

    tensor_to_stlnet(test_predicted_traces, predicted_stlnet_file0)
    stlnet_to_traces(predicted_stlnet_file0, predicted_traces_file0, alphabet)

    # Convert test dataset to files
    tensor_to_stlnet(test_predicted_traces, predicted_stlnet_file)
    stlnet_to_traces(predicted_stlnet_file, predicted_traces_file, alphabet)
    traces_to_xes(predicted_traces_file, predicted_log_file)

    # Align and convert back
    align_traces(predicted_log_file, form_file, predicted_traces_file, max_length=trace_length)
    discard_transition(predicted_traces_file)
    standardize_trace_length(predicted_traces_file)
    traces_to_stlnet(predicted_traces_file, predicted_stlnet_file, alphabet)
    
    # Load aligned test dataset
    test_predicted_traces = torch.tensor(np.loadtxt(predicted_stlnet_file))
    test_predicted_traces = test_predicted_traces.view(test_predicted_traces.size(0), -1, len(alphabet))
    test_predicted_traces = test_predicted_traces.float()

    #os.remove(predicted_log_file)
    #os.remove(predicted_traces_file)
    #os.remove(predicted_stlnet_file)

    return test_predicted_traces
