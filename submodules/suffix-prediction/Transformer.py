import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from FiniteStateMachine import DFA

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class TransformerModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size, tagset_size, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_size = vocab_size
        
        # Embedding layer - converts one-hot encoding to dense embeddings
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        
        # Positional encoding to give the model information about position of tokens
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Create a transformer decoder layer with causal self-attention
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        
        # Create the decoder stack
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, tagset_size)
        
        # Activation for positivity
        self.positive_activation = nn.ReLU()
        
        # Cache to store the causal mask
        self.causal_mask = None
        
    def _generate_square_subsequent_mask(self, sz):
        """Generate a causal mask for the decoder's self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Create causal attention mask
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = self._generate_square_subsequent_mask(seq_len)
        
        # Embed the input
        x_embedded = self.embedding(x)
        
        # Add positional encoding
        x_embedded = self.pos_encoder(x_embedded)
        
        transformer_output = self.transformer_decoder(
            tgt=x_embedded,                          
            memory=torch.zeros_like(x_embedded),     
            tgt_mask=self.causal_mask,               
            memory_mask=None,                        
            tgt_key_padding_mask=None,               
            memory_key_padding_mask=None             
        )
        
        # Output layer
        tag_space = self.positive_activation(self.output_layer(transformer_output)) + 0.5
        
        # Return output and the transformer output as state (for continuation)
        return tag_space, transformer_output
    
    def forward_from_state(self, x, state):        
        # Get batch size and the current sequence length from state
        batch_size = x.size(0)
        
        # Embed the new token
        x_embedded = self.embedding(x)
        
        # Append the new token embedding to the previous context
        # This creates a growing context that the model can attend to
        full_context = torch.cat([state, x_embedded], dim=1)
        seq_len = full_context.size(1)
        
        # Update positional encoding for the full sequence
        full_context = self.pos_encoder(full_context, is_continuation=True)
        
        # Ensure causal mask is large enough
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            self.causal_mask = self._generate_square_subsequent_mask(seq_len)
        
        # Process through decoder with causal masking
        # We only need to compute the last position
        transformer_output = self.transformer_decoder(
            tgt=full_context,
            memory=torch.zeros_like(full_context),
            tgt_mask=self.causal_mask[:seq_len, :seq_len],
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # Extract only the prediction for the last token
        last_token_output = transformer_output[:, -1:, :]
        tag_space = self.positive_activation(self.output_layer(last_token_output)) + 0.5
        
        # Return prediction for the last token and the updated state
        return tag_space, full_context
    
    def next_sym_prob(self, x, state):
        tag_space, state = self.forward_from_state(x, state)
        tag_space = F.softmax(tag_space, dim=-1)
        return tag_space, state
    
    def predict(self, sentence):
        tag_space, _ = self.forward(sentence)
        out = F.softmax(tag_space[:, -1, :], dim=-1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to store positional encodings
        self.register_buffer('pe', pe)
        self.position_counter = 0
        
    def forward(self, x, is_continuation=False):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            is_continuation: Boolean indicating if this is continuing from a previous sequence
        """
        seq_len = x.size(1)
        
        if is_continuation:
            # For continuation, maintain position counter to ensure proper positions
            pos_start = self.position_counter
            self.position_counter += 1  # Only increment by 1 for the new token
        else:
            # Reset counter for new sequences
            pos_start = 0
            self.position_counter = seq_len
            
        # Add positional encoding
        x = x + self.pe[pos_start:pos_start+seq_len].unsqueeze(0)
        return self.dropout(x)


class TransformerWithConstraints(nn.Module):
    def __init__(self, transformer, ltl_formula):
        super(TransformerWithConstraints, self).__init__()
        
        self.transformer = transformer
        # Formula evaluator
        dfa = DFA(ltl_formula, 2, "random DNF declare", ['c0', 'c1', 'end'])
        self.deep_dfa_constraint = dfa.return_deep_dfa_constraint()
        
    def forward(self, x):
        pred_sym, hidden_states = self.transformer(x)
        
        # Transform one-hot into indices
        x_indices = torch.argmax(x, dim=-1).long()
        
        # Get masks from DFA
        masks, dfa_state = self.deep_dfa_constraint(x_indices)
        
        # Apply constraints
        pred_sym = pred_sym * masks
        
        return pred_sym, (hidden_states, dfa_state)
        
    def forward_from_state(self, x, tot_state):
        state_transformer, state_dfa = tot_state
        
        next_event, next_state_transformer = self.transformer.forward_from_state(x, state_transformer)
        
        next_event = next_event.squeeze()
        x = torch.argmax(x, -1).squeeze()
        next_dfa_state, mask = self.deep_dfa_constraint.step(state_dfa, x)
        
        next_event = next_event * mask
        
        return next_event.unsqueeze(1), (next_state_transformer, next_dfa_state)