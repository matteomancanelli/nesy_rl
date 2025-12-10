import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        logits = self.output_layer(output)
        return logits, (hn, cn)

    def forward_from_state(self, x, state):
        output, (hn, cn) = self.lstm(x, state)
        logits = self.output_layer(output)
        return logits, (hn, cn)


class LogicLoss(nn.Module):
    def __init__(self, dfa, alpha):
        super().__init__()
        self.dfa = dfa
        self.alpha = alpha

    def forward(self, predictions, targets, inputs, device):
        probs = F.softmax(predictions, dim=-1)
        batch_size, seq_len, vocab_size = predictions.shape

        # DFA transitions
        token_indices = torch.argmax(inputs, dim=-1)
        state_ids = self.dfa.simulate(token_indices, batch_size)

        dfa_state_indices_exp = state_ids.unsqueeze(-1).expand(-1, -1, vocab_size)
        symbol_indices = torch.arange(vocab_size).to(device).view(1, 1, vocab_size).expand_as(dfa_state_indices_exp)
        next_states = self.dfa.transition_tensor[dfa_state_indices_exp, symbol_indices]

        # Cross-Entropy Loss per step
        ce_loss_per_step = F.cross_entropy(
            predictions.view(-1, vocab_size), targets.view(-1), reduction='none'
        ).view(batch_size, seq_len)

        gather_indices = targets.unsqueeze(-1)
        state_importance = (self.dfa.state_types_tensor[next_states] != -1).float()  # 1.0 for valid, 0.0 for invalid
        soft_weights = torch.where(state_importance > 0, 1.0, 0.05)  # 0.05 for failure states
        ce_weights = torch.gather(soft_weights, 2, gather_indices).squeeze(-1)

        weighted_ce_loss = (ce_loss_per_step * ce_weights).sum() / (ce_weights.sum() + 1e-6)

        # Symbolic Loss
        reject_mask = self.dfa.state_types_tensor[next_states] == -1
        invalid_mass = (probs * reject_mask.float()).sum(dim=-1).mean()
        step_penalty = -torch.log(1.0 - invalid_mass + 1e-6)

        return self.alpha * weighted_ce_loss + (1 - self.alpha) * step_penalty
