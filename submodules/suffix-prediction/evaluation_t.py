import torch
from statistics import mean
from torch.nn.utils.rnn import pad_sequence
if torch.cuda.is_available():
     device = 'cuda:0'
else:
    device = 'cpu'

cross_entr_func = torch.nn.CrossEntropyLoss()

def evaluate_accuracy_next_activity(model, test_dataset, acc_func):
    model = model.to(device)
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for batch in [test_dataset]:
            X = batch[:, :-1, :].to(device)
            Y = batch[:, 1:, :]
            target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)
            
            predictions, _ = model(X)
            predictions = predictions.reshape(-1, predictions.size()[-1])
            
            accuracies.append(acc_func(predictions, target).item())
    
    return mean(accuracies)

import torch.nn.functional as F

def round_to_one_hot(tensor):
    # Find the index of the maximum value along the last dimension
    max_indices = torch.argmax(tensor, dim=-1, keepdim=True)

    # Create a one-hot tensor with the same shape as the input tensor
    one_hot_tensor = torch.zeros_like(tensor)

    # Set the element corresponding to the maximum value to 1
    one_hot_tensor.scatter_(-1, max_indices, 1)

    return one_hot_tensor

def is_transformer_model(model):
    """Helper function to check if a model is a Transformer"""
    return 'Transformer' in model.__class__.__name__

def greedy_suffix_prediction(model, dataset, prefix_len):
    """
    Generate a suffix using greedy prediction, adapted for both RNN and Transformer models.
    """
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    
    predicted_traces = prefix
    len_traces = dataset.size()[1]
    
    # Get first prediction and state
    next_event, model_state = model(prefix)
    
    # Check if using transformer
    is_transformer = is_transformer_model(model)
    
    for step in range(prefix_len, len_traces*2):
        next_event = F.softmax(next_event[:, -1:, :], dim=-1)
        next_event = round_to_one_hot(next_event)
        
        predicted_traces = torch.cat((predicted_traces, next_event), dim=1)
        
        # For transformer, we might need to manage context differently
        next_event, model_state = model.forward_from_state(next_event, model_state)
    
    return predicted_traces

def sample_with_temperature(probabilities, temperature=1.0):
    """
    Sample from a probability distribution with temperature control.
    """
    if temperature == 0:
        return torch.argmax(probabilities, dim=-1)
    else:
        batch_size = probabilities.size()[0]
        num_classes = probabilities.size()[-1]
        num_samples = 1
        
        # Add small epsilon to avoid numerical issues
        probabilities = probabilities + 1e-10
        indices = torch.multinomial(probabilities.squeeze(), num_samples)
        
        # Create one-hot vectors based on the drawn indices
        one_hot_vectors = torch.zeros(batch_size, num_samples, num_classes).to(device)
        one_hot_vectors.scatter_(2, indices.unsqueeze(-1), 1)
        
        return one_hot_vectors

def suffix_prediction_with_temperature(model, dataset, prefix_len, temperature=1.0):
    """
    Generate a suffix using temperature-based sampling, adapted for both RNN and Transformer models.
    """
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    
    predicted_traces = prefix
    len_traces = dataset.size()[1]
    
    next_event, model_state = model(prefix)
    
    for step in range(prefix_len, len_traces):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = sample_with_temperature(next_event, temperature)
        
        predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)
        
        next_event, model_state = model.forward_from_state(next_event_one_hot, model_state)
    
    return predicted_traces

def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    """
    Gumbel-Softmax sampling function.
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def logic_loss(model, deepdfa, data, prefix_len, temperature=1.0):
    """
    Calculate logic loss, adapted for both RNN and Transformer models.
    """
    dataset = data.to(device)
    prefix = dataset[:, :prefix_len, :]
    
    batch_size = dataset.size()[0]
    target = torch.ones(batch_size, dtype=torch.long, device=device)
    
    len_traces = dataset.size()[1]
    next_event, model_state = model(prefix)
    dfa_states, dfa_rew = deepdfa.forward_pi(prefix)
    
    dfa_state = dfa_states[:, -1, :]
    
    for step in range(prefix_len, int(len_traces*(1.5))):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = gumbel_softmax(next_event, temperature)
        
        # Transit on the automaton
        dfa_state, dfa_rew = deepdfa.step_pi(dfa_state, next_event_one_hot.squeeze())
        
        next_event, model_state = model.forward_from_state(next_event_one_hot, model_state)
    
    loss = cross_entr_func(100*dfa_rew, target)
    return loss

def logic_loss_multiple_samples(model, deepdfa, data, prefix_len, temperature=1.0, num_samples=10):
    """
    Calculate logic loss with multiple samples, adapted for both RNN and Transformer models.
    """
    dataset = data.to(device)
    prefix = dataset[:, :prefix_len, :]
    
    batch_size, len_traces, num_activities = dataset.size()
    target = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # Extend prefix
    prefix = prefix.unsqueeze(1).repeat(1, num_samples, 1, 1).view(-1, prefix_len, num_activities)
    
    # Calculate next symbol and dfa state
    next_event, model_state = model(prefix)
    dfa_states, dfa_rew = deepdfa.forward_pi(prefix)
    
    dfa_state = dfa_states[:, -1, :]
    
    log_prob_traces = torch.zeros((batch_size*num_samples, 1)).to(device)
    for step in range(prefix_len, int(len_traces*(1.5))):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = gumbel_softmax(next_event, temperature)
        
        log_prob_traces += torch.sum(next_event * next_event_one_hot, dim=-1)
        
        # Transit on the automaton
        dfa_state, dfa_rew = deepdfa.step_pi(dfa_state, next_event_one_hot.squeeze())
        
        # Transit the model
        next_event, model_state = model.forward_from_state(next_event_one_hot, model_state)
    
    dfa_rew = dfa_rew.view(batch_size, num_samples, 2)
    dfa_rew = dfa_rew[:,:,1]
    log_prob_traces = log_prob_traces.view(batch_size, num_samples)
    
    prob_acceptance = torch.sum(torch.nn.functional.softmax(log_prob_traces, dim=-1) * dfa_rew, dim=-1)
    
    loss = -torch.log(prob_acceptance.clamp(min=1e-10)).mean()
    return loss

def suffix_prediction_with_temperature_with_stop(model, dataset, prefix_len, stop_event=[0,0,0,1], temperature=1.0):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    
    predicted_traces = prefix
    len_traces = dataset.size()[1]
    
    next_event, model_state = model(prefix)
    
    # Initialize a mask indicating which sequences have reached the stop event
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)
    
    for step in range(prefix_len, len_traces *2):
        next_event = next_event[:, -1:, :]
        next_event_one_hot = sample_with_temperature(next_event, temperature)
        
        predicted_traces = torch.cat((predicted_traces, next_event_one_hot), dim=1)
        
        # Check if any sequence has reached the stop event and update the stop mask
        stop_mask |= torch.all(next_event_one_hot.squeeze() == torch.tensor(stop_event).to(device), dim=-1)
        
        # Check if all sequences have reached the stop event
        if torch.all(stop_mask):
            break  # Stop predicting if all sequences have reached the stop event
        
        next_event, model_state = model.forward_from_state(next_event_one_hot, model_state)
    
    return predicted_traces

def greedy_suffix_prediction_with_stop(model, dataset, prefix_len, stop_event=[0,0,0,1]):
    """
    Generate a suffix with greedy prediction until a stop event is reached.
    Adapted for both RNN and Transformer models.
    """
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    
    predicted_traces = prefix
    len_traces = dataset.size()[1]
    
    next_event, model_state = model(prefix)
    
    # Initialize a mask indicating which sequences have reached the stop event
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)
    
    for step in range(prefix_len, len_traces*2):
        next_event = F.softmax(next_event[:, -1:, :], dim=-1)
        next_event = round_to_one_hot(next_event)
        
        predicted_traces = torch.cat((predicted_traces, next_event), dim=1)
        
        # Check if any sequence has reached the stop event and update the stop mask
        stop_mask |= torch.all(next_event.squeeze() == torch.tensor(stop_event).to(device), dim=-1)
        
        # Check if all sequences have reached the stop event
        if torch.all(stop_mask):
            break  # Stop predicting if all sequences have reached the stop event
        
        next_event, model_state = model.forward_from_state(next_event, model_state)
    
    return predicted_traces

def evaluate_compliance_with_formula(deepdfa, traces):
    """
    Evaluate how well the traces comply with a formula.
    """
    traces = torch.argmax(traces, dim=-1)
    
    r, _ = deepdfa(traces)
    accepted = r[:,-1,-1]
    
    return accepted.mean().item()

def evaluate_DL_distance(predicted_traces, target_traces):
    """
    Evaluate the Damerau-Levenshtein distance between predicted and target traces.
    """
    DL_dists = []
    
    for i in range(predicted_traces.size()[0]):
        pred = tensor_to_string(predicted_traces[i])
        targ = tensor_to_string(target_traces[i])
        DL_dists.append(damerau_levenshtein_distance(pred, targ))
    
    return mean(DL_dists)

def tensor_to_string(one_hot_tensor):
    """
    Convert a one-hot tensor to a string representation.
    """
    end_symbol = one_hot_tensor.size()[-1] - 1
    
    # Convert the one-hot tensor to a numpy array
    numpy_array = one_hot_tensor.cpu().numpy()
    
    # Extract indices of maximum values along the second dimension
    indices = numpy_array.argmax(axis=1)
    
    # Convert indices into a string
    string = ''
    for idx in indices:
        string += str(idx)
    
    return string

def damerau_levenshtein_distance(str1, str2):
    """
    Calculate the Damerau-Levenshtein distance between two strings.
    """
    len_str1 = len(str1)
    len_str2 = len(str2)
    # Create a matrix to store the distances between substrings
    matrix = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    # Initialize the first row and column of the matrix
    for i in range(len_str1 + 1):
        matrix[i][0] = i
    for j in range(len_str2 + 1):
        matrix[0][j] = j

    # Populate the matrix
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(matrix[i-1][j] + 1,          # Deletion
                               matrix[i][j-1] + 1,          # Insertion
                               matrix[i-1][j-1] + cost)    # Substitution
            # Check for transposition
            if i > 1 and j > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
                matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2] + cost)

    return matrix[len_str1][len_str2]