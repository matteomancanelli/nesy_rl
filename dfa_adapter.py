import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party_shims"))

import suffix_prediction  # type: ignore


class TTDFAAdapter:
    def __init__(
        self, observation_dim, action_dim, num_bins,
        include_reward=True, include_value=True,
        constraint_dims=None, abstraction_fn=None, use_stop_token=True
    ):
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

        self.constraint_dims = constraint_dims
        self.abstraction_fn = abstraction_fn
        self.use_stop_token = use_stop_token

        total_dims = self.observation_dim + self.action_dim
        if include_reward:
            total_dims += 1
        if include_value:
            total_dims += 1

        if isinstance(num_bins, int):
            self.num_bins_per_dim = [int(num_bins)] * total_dims
        elif isinstance(num_bins, (list, tuple)):
            if len(num_bins) != total_dims:
                raise ValueError(f"len(num_bins)={len(num_bins)} but total_dims={total_dims}")
            self.num_bins_per_dim = [int(b) for b in num_bins]
        else:
            raise ValueError("num_bins must be int or list/tuple of ints")

        if any(b <= 0 for b in self.num_bins_per_dim):
            raise ValueError("All num_bins entries must be positive")

        self.transition_dim = total_dims
        self.max_num_bins = max(self.num_bins_per_dim)
        self.num_token_ids = self.max_num_bins + (1 if self.use_stop_token else 0)

        # expose stop id for logic_loss_tt.py
        self.stop_token_id = self.max_num_bins if self.use_stop_token else None

        self.r_idx = observation_dim + action_dim if include_reward else None
        self.v_idx = observation_dim + action_dim + (1 if include_reward else 0) if include_value else None

        self.symbolic_vocab = []
        self.symbol_to_idx = {}
        self.pos_bin_to_sym_idx = torch.empty(self.transition_dim, self.num_token_ids, dtype=torch.long)

        self._build_symbolic_vocab_and_mapping()
        self.num_symbols = len(self.symbolic_vocab)

    def _global_dim_for_pos(self, pos: int) -> int:
        return pos

    def _gen_symbol_name(self, global_dim: int, bin_id: int) -> str:
        if self.use_stop_token and bin_id == self.max_num_bins:
            return "end"

        if global_dim < self.observation_dim:
            prefix = f"s{global_dim}"
        elif global_dim < self.observation_dim + self.action_dim:
            prefix = f"a{global_dim - self.observation_dim}"
        elif self.r_idx is not None and global_dim == self.r_idx:
            prefix = "r"
        elif self.v_idx is not None and global_dim == self.v_idx:
            prefix = "v"
        else:
            prefix = f"x{global_dim}"

        if self.abstraction_fn is not None and bin_id < self.max_num_bins:
            label = self.abstraction_fn(global_dim, bin_id)
            return f"{prefix}_{label}"
        return f"{prefix}_bin{bin_id}"

    def _add_symbol(self, symbol_name: str) -> int:
        if symbol_name not in self.symbol_to_idx:
            idx = len(self.symbolic_vocab)
            self.symbolic_vocab.append(symbol_name)
            self.symbol_to_idx[symbol_name] = idx
        return self.symbol_to_idx[symbol_name]

    def _build_symbolic_vocab_and_mapping(self):
        for pos in range(self.transition_dim):
            global_dim = self._global_dim_for_pos(pos)
            for bin_id in range(self.num_token_ids):
                sym = self._gen_symbol_name(global_dim, bin_id)
                sym_idx = self._add_symbol(sym)
                self.pos_bin_to_sym_idx[pos, bin_id] = sym_idx

    def tokens_to_symbols(self, tokens: torch.Tensor):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.long()
        B, T = tokens.shape

        # vectorized mapping tokens->symbol_idx, then symbol_idx->string
        pos = torch.arange(T, device=tokens.device) % self.transition_dim  # [T]
        mapping = self.pos_bin_to_sym_idx.to(tokens.device)[pos]            # [T, V]
        tok_clamped = tokens.clamp(min=0, max=self.num_token_ids - 1)       # [B, T]
        sym_idx = mapping.gather(1, tok_clamped.transpose(0, 1)).transpose(0, 1)  # [B, T]

        out = []
        for b in range(B):
            out.append([self.symbolic_vocab[int(i)] for i in sym_idx[b]])
        return out[0] if B == 1 else out

    def token_probs_to_symbol_probs(self, token_probs: torch.Tensor) -> torch.Tensor:
        """
        token_probs: [B, T, V=num_token_ids]
        sym_probs:   [B, T, S=num_symbols]
        Vectorized (no Python loop over T).
        """
        if token_probs.size(-1) != self.num_token_ids:
            raise ValueError(f"token_probs last dim {token_probs.size(-1)} != num_token_ids {self.num_token_ids}")

        B, T, V = token_probs.shape
        device = token_probs.device

        pos = (torch.arange(T, device=device) % self.transition_dim)          # [T]
        mapping = self.pos_bin_to_sym_idx.to(device)[pos]                     # [T, V]
        mapping_bt = mapping.unsqueeze(0).expand(B, T, V).reshape(B * T, V)    # [B*T, V]

        tp = token_probs.reshape(B * T, V)
        sym_probs = tp.new_zeros((B * T, self.num_symbols))
        sym_probs.scatter_add_(1, mapping_bt, tp)

        return sym_probs.reshape(B, T, self.num_symbols)

    def create_dfa_from_ltl(self, ltl_formula, formula_name="constraint"):
        # Lazy import to avoid segfault on import in some envs
        from FiniteStateMachine import DFA
        return DFA(ltl_formula, self.num_symbols, formula_name, self.symbolic_vocab)

    # keep your batch_check_dfa_sat for now (eval-only); it’s slower but not the training bottleneck
    def _symbols_to_dfa_indices(self, symbol_seq, dfa):
        indices = []
        for symbol in symbol_seq:
            if symbol in dfa.dictionary_symbols:
                idx = dfa.dictionary_symbols.index(symbol)
            else:
                idx = dfa.dictionary_symbols.index("end") if "end" in dfa.dictionary_symbols else 0
            indices.append(idx)
        return indices

    def batch_check_dfa_sat(self, token_sequences, dfa, device="cuda:0"):
        if token_sequences.dim() != 2:
            raise ValueError("batch_check_dfa_sat expects [batch_size, seq_len]")

        token_sequences = token_sequences.long()
        B = token_sequences.shape[0]
        satisfaction = torch.zeros(B, dtype=torch.float32, device=device)

        symbol_sequences = self.tokens_to_symbols(token_sequences)
        for i, symbol_seq in enumerate(symbol_sequences):
            indices = self._symbols_to_dfa_indices(symbol_seq, dfa)
            satisfaction[i] = 1.0 if dfa.accepts_from_state(0, indices) else 0.0

        return satisfaction
