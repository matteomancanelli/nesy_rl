import torch
import torch.nn.functional as F

class LogicLossModule:
    """
    Logic-aware loss for Trajectory Transformer with Deep DFA constraints.

    Combines:
      - supervised action/token prediction loss (masked)
      - global logic loss based on DeepDFA acceptance over generated sequences

    Total:
      total_loss = (1 - alpha) * sup_loss + alpha * logic_loss
    """

    def __init__(
        self,
        deep_dfa,
        adapter,
        mode="global",
        num_samples=10,
        temperature=0.5,
        alpha=0.4,
        eps=1e-10,
    ):
        self.deep_dfa = deep_dfa
        self.adapter = adapter
        self.mode = mode
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def _masked_cross_entropy(self, logits, targets, mask):
        """
        logits:  [B, T, V]
        targets: [B, T]
        mask:    [B, T]  (1 for valid positions, 0 for padding)
        """
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        mask_flat = mask.reshape(B * T).float()

        loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        denom = mask_flat.sum().clamp(min=1.0)
        return (loss_flat * mask_flat).sum() / denom

    def _gumbel_softmax_samples(self, logits):
        """
        logits: [B, T, V]
        Returns:
          samples:   [B, S, T, V]  (soft one-hots)
          log_probs: [B, T, V]
        """
        B, T, V = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        logits_exp = logits.unsqueeze(1).expand(B, self.num_samples, T, V)
        samples = F.gumbel_softmax(
            logits_exp, tau=self.temperature, hard=False, dim=-1
        )
        return samples, log_probs

    def _apply_stop_token_on_padding(self, samples, mask, stop_token_id):
        """
        Force padded positions to be a deterministic STOP token distribution so
        the DFA sees termination instead of arbitrary sampled tokens.

        samples: [B, S, T, V]
        mask:    [B, T] (1 valid, 0 pad)
        """
        if stop_token_id is None:
            return samples

        B, S, T, V = samples.shape
        dev = samples.device

        # mask_valid: [B, 1, T, 1]
        mask_valid = mask.to(dev).unsqueeze(1).unsqueeze(-1).float()

        # one-hot stop: [1, 1, 1, V]
        stop = torch.zeros((1, 1, 1, V), device=dev)
        stop[..., stop_token_id] = 1.0

        # Where mask==0 => replace with stop distribution
        return samples * mask_valid + stop * (1.0 - mask_valid)

    def global_logic_loss_tt(self, model, batch, return_components=False):
        """
        batch = (X, Y, mask)
          X:    model inputs
          Y:    token targets for supervised loss
          mask: [B, T] valid positions
        """
        if len(batch) != 3:
            raise ValueError(f"Expected batch to be (X, Y, mask); got length {len(batch)}")

        x, y, mask = batch

        # Forward pass
        logits, sup_loss = model(x, targets=y, mask=mask)

        B, T, V = logits.shape
        dev = logits.device

        x = x.to(dev)
        y = y.to(dev)
        mask = mask.to(dev)

        if V != self.adapter.num_token_ids:
            raise ValueError(
                f"Model logits last dim ({V}) != adapter.num_token_ids ({self.adapter.num_token_ids})"
            )

        # 1) Supervised loss (masked)
        sup_loss = self._masked_cross_entropy(logits, y, mask)

        # 2) Gumbel-softmax samples
        samples, _ = self._gumbel_softmax_samples(logits)  # [B, S, T, V]

        # 3) Ensure padding positions correspond to STOP token for DFA purposes
        stop_token_id = getattr(self.adapter, "stop_token_id", None)
        samples = self._apply_stop_token_on_padding(samples, mask, stop_token_id)

        # 4) Map token distributions -> DFA symbol distributions
        traces_soft = samples.reshape(B * self.num_samples, T, V)
        sym_probs = self.adapter.token_probs_to_symbol_probs(traces_soft).to(dev)

        # 5) DeepDFA acceptance
        deep_dfa = self.deep_dfa.to(dev)
        _, dfa_rew_seq = deep_dfa.forward_pi(sym_probs)
        dfa_final = dfa_rew_seq[:, -1, :]

        if dfa_final.size(-1) < 2:
            raise ValueError("DeepDFA final reward has <2 outputs; expected [reject, accept].")

        acceptance = dfa_final[:, 1].reshape(B, self.num_samples)

        # 6) Monte Carlo estimate (no importance weights)
        prob_acceptance = acceptance.mean(dim=1)  # [B]

        logic_loss = -torch.log(prob_acceptance.clamp(min=self.eps)).mean()

        total_loss = (1.0 - self.alpha) * sup_loss + self.alpha * logic_loss

        if return_components:
            return total_loss, sup_loss, logic_loss
        return total_loss

    def local_logic_loss_tt(self, *args, **kwargs):
        raise NotImplementedError(
            "Local logic loss for TT is not implemented yet. Use mode='global'."
        )

    def compute_loss(self, model, batch, return_components=False):
        if self.mode == "global":
            return self.global_logic_loss_tt(model, batch, return_components=return_components)
        if self.mode == "local":
            return self.local_logic_loss_tt()
        raise ValueError(f"Unknown mode: {self.mode}. Use 'global' or 'local'.")
