# core/runner.py
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

from helpers.checkpointing import save_checkpoint, load_checkpoint
from helpers.io import save_json, save_npz
from logic_loss_tt import LogicLossModule

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party_shims"))

import trajectory_transformer # type: ignore
from trajectory.models.transformers import GPT


def build_model(args, dataset, vocab_size: int, device: torch.device):
    class Cfg:
        pass

    cfg = Cfg()
    cfg.vocab_size = vocab_size
    cfg.block_size = args.block_size
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    cfg.observation_dim = dataset.observation_dim
    cfg.action_dim = dataset.action_dim
    cfg.transition_dim = dataset.joined_dim
    cfg.action_weight = args.action_weight
    cfg.reward_weight = args.reward_weight
    cfg.value_weight = args.value_weight
    cfg.embd_pdrop = args.embd_pdrop
    cfg.resid_pdrop = args.resid_pdrop
    cfg.attn_pdrop = args.attn_pdrop

    model = GPT(cfg).to(device)
    return model, cfg


@torch.no_grad()
def eval_model(
    model,
    adapter,
    raw_dfa,
    loader,
    device: torch.device,
    save_traces_path: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_sup = 0.0
    n_batches = 0

    total_sat = 0.0
    total_seq = 0

    # optional: store some predictions
    stored_preds = []
    stored_sat = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        x, y, mask = [t.to(device) for t in batch]
        logits, sup_loss = model(x, targets=y, mask=mask)
        preds = logits.argmax(dim=-1)  # [B, T]

        if isinstance(raw_dfa, (list, tuple)):
            sats = [adapter.batch_check_dfa_sat(preds, d) for d in raw_dfa]
            sat = torch.stack(sats, dim=0).min(dim=0).values
        else:
            sat = adapter.batch_check_dfa_sat(preds, raw_dfa)

        total_sup += float(sup_loss.item())
        n_batches += 1

        total_sat += float(sat.sum().item())
        total_seq += int(sat.numel())

        if save_traces_path is not None and len(stored_preds) < 200:
            stored_preds.append(preds.detach().cpu())
            stored_sat.append(sat.detach().cpu())

    metrics = {
        "supervised_loss": total_sup / max(1, n_batches),
        "satisfaction_rate": total_sat / max(1, total_seq),
        "violation_rate": 1.0 - (total_sat / max(1, total_seq)),
    }

    if save_traces_path is not None:
        preds_cat = torch.cat(stored_preds, dim=0) if stored_preds else torch.empty(0, dtype=torch.long)
        sat_cat = torch.cat(stored_sat, dim=0) if stored_sat else torch.empty(0, dtype=torch.float32)
        save_npz(save_traces_path, preds=preds_cat, sat=sat_cat)

    return metrics


def train_one_run(
    args,
    benchmark_assets,
    out_dir: str,
    alpha: float,
    device: torch.device,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    dataset = benchmark_assets.dataset
    adapter = benchmark_assets.adapter
    deep_dfa = benchmark_assets.deep_dfa
    raw_dfa = benchmark_assets.raw_dfa

    # IMPORTANT: your adapter reserves stop token as max_bin_id, but TT GPT expects vocab_size
    # without that extra stop if you used that convention before.
    # You were doing: vocab_size = adapter.num_token_ids - 1
    model, model_cfg = build_model(args, dataset, vocab_size=adapter.num_token_ids - 1, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # logic module
    # Assumption: LogicLossModule exposes eps/clamp_acceptance in your current version.
    logic = LogicLossModule(
        deep_dfa=deep_dfa,
        adapter=adapter,
        mode="global",
        num_samples=args.num_samples,
        temperature=args.temperature,
        alpha=alpha,
        eps=getattr(args, "logic_eps", 1e-10)
    )

    # split dataset for periodic eval
    n = len(dataset)
    n_val = int(args.val_ratio * n)
    n_test = int(args.test_ratio * n)
    n_train = max(1, n - n_val - n_test)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    # resume support
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        last_epoch, extra = load_checkpoint(
            ckpt_path, model, opt,
            map_location=str(device),
            restore_rng=True,
        )
        start_epoch = last_epoch + 1

    # save run config
    save_json(os.path.join(out_dir, "run_config.json"), {
        "alpha": alpha,
        "benchmark": args.benchmark,
        "ltl_formula": args.ltl_formula,
        "ltl_formulas": args.ltl_formulas,
        "dfa_mode": args.dfa_mode,
        "train": {k: getattr(args, k) for k in vars(args)},
        "model_cfg": {k: getattr(model_cfg, k) for k in dir(model_cfg) if not k.startswith("__")},
    })

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        sup = 0.0
        logl = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = [t.to(device) for t in batch]
            x, y, mask = batch

            if alpha == 0.0:
                # TRUE vanilla: supervised only (fast)
                logits, sup_loss = model(x, targets=y, mask=mask)
                loss = sup_loss
                logic_loss = torch.tensor(0.0, device=device)
            else:
                loss, sup_loss, logic_loss = logic.compute_loss(model, batch, return_components=True)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total += float(loss.item())
            sup += float(sup_loss.item())
            logl += float(logic_loss.item())
            n_batches += 1

        train_row = {
            "epoch": epoch,
            "train_total_loss": total / max(1, n_batches),
            "train_sup_loss": sup / max(1, n_batches),
            "train_logic_loss": logl / max(1, n_batches),
        }

        # periodic eval + trace dump
        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            val_metrics = eval_model(
                model, adapter, raw_dfa, val_loader, device,
                save_traces_path=os.path.join(out_dir, f"val_preds_epoch{epoch}.npz") if args.save_traces else None,
                max_batches=args.eval_max_batches,
            )
            train_row.update({f"val_{k}": v for k, v in val_metrics.items()})

        history.append(train_row)
        save_json(os.path.join(out_dir, "history.json"), {"rows": history})

        # checkpoint EVERY epoch
        save_checkpoint(
            ckpt_path,
            model,
            opt,
            epoch=epoch,
            extra={"history_len": len(history)},
        )

        print(
            f"[{os.path.basename(out_dir)}] epoch {epoch} | "
            f"loss={train_row['train_total_loss']:.4f} "
            f"sup={train_row['train_sup_loss']:.4f} "
            f"logic={train_row['train_logic_loss']:.4f} "
            + (f"| val_sat={train_row.get('val_satisfaction_rate', float('nan')):.3f}" if "val_satisfaction_rate" in train_row else "")
        )

    # final test eval (optionally later)
    test_metrics = eval_model(
        model, adapter, raw_dfa, test_loader, device,
        save_traces_path=os.path.join(out_dir, "test_preds_final.npz") if args.save_traces else None,
        max_batches=None,
    )
    save_json(os.path.join(out_dir, "final_metrics.json"), {"test": test_metrics})

    return {"out_dir": out_dir, "test": test_metrics}
