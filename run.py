# core/runner.py
import os
from pathlib import Path
import sys
import numpy as np
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


@torch.no_grad()
def rollout_eval_cb(
    model,
    env,
    adapter,
    device: torch.device,
    n_episodes: int = 50,
    max_steps: int = None,
    model_kind: str = "tt",
    v_bins: int = 1,
    dt_target_vtoken: int = None,
):
    """
    Closed-loop rollout evaluation for Colour Bomb.

    Important: your model is trained to predict the next *row* (shift by transition_dim).
    So to predict action a_t, we read logits at the position corresponding to a_{t-1}.
    For t=0 we bootstrap with a dummy previous row.
    """
    model.eval()
    if max_steps is None:
        max_steps = env.cfg.max_steps

    # choose conditioning v token
    if model_kind == "dt":
        if dt_target_vtoken is None:
            dt_target_vtoken = max(0, int(v_bins) - 1)
        v_tok = int(dt_target_vtoken)
    else:
        v_tok = 0

    returns = []
    sats = []

    stop_id = getattr(adapter, "stop_token_id", None)
    if stop_id is None:
        stop_id = adapter.max_num_bins  # fallback

    for ep in range(n_episodes):
        s, _ = env.reset()
        ep_ret = 0.0

        # history of rows [s,a,r,v] as integers
        rows = []

        # bootstrap "previous row" so we can query a_0 from logits at a_{-1}
        # pick a dummy prev action 0 and reward token 0 (your datasets use r=0 tokens anyway)
        rows.append([int(s), 0, 0, v_tok])

        done = False
        for t in range(max_steps):
            # build context tokens
            flat = np.array(rows, dtype=np.int64).reshape(-1)
            # truncate to last block_size tokens if needed
            # (model cfg block_size is args.block_size; we’ll just keep last adapter-aligned suffix)
            x = torch.from_numpy(flat).to(device).long()
            if x.numel() > model.cfg.block_size:
                x = x[-model.cfg.block_size:]

            # mask/targets are dummy (we only need logits)
            mask = torch.ones_like(x, dtype=torch.float32, device=device)
            # make a fake y of same length (not used)
            y = torch.zeros_like(x, device=device)

            logits, _ = model(x.unsqueeze(0), targets=y.unsqueeze(0), mask=mask.unsqueeze(0))

            # find index of "a_{t-1}" in the current x
            # rows[-1] is the previous row; action is position 1 within row
            # in flat history, index of previous action is (len(rows)-1)*4 + 1
            idx_in_full = (len(rows) - 1) * adapter.transition_dim + 1
            # after truncation, shift
            idx_in_x = idx_in_full - (flat.size - x.numel())
            idx_in_x = int(np.clip(idx_in_x, 0, x.numel() - 1))

            a_logits = logits[0, idx_in_x, :]  # [V]
            a = int(torch.argmax(a_logits).item())

            # step env with predicted action
            ns, r, done, _info = env.step(a)
            ep_ret += float(r)

            # append the executed transition row (reward token still 0)
            rows.append([int(ns), int(a), 0, int(v_tok)])
            s = ns
            if done:
                break

        # build token sequence and check DFA sat (append stop row)
        tok = np.array(rows, dtype=np.int64)
        end_row = np.array([[stop_id] * adapter.transition_dim], dtype=np.int64)
        tok = np.vstack([tok, end_row])
        flat_tok = torch.from_numpy(tok.reshape(-1)).unsqueeze(0).to(device)

        if hasattr(env, "cfg"):
            pass  # placeholder; env exists

        # raw_dfa may be list; caller will handle. Here we just return the trace.
        returns.append(ep_ret)
        sats.append(flat_tok)

    return returns, sats


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

    model, model_cfg = build_model(args, dataset, vocab_size=adapter.num_token_ids, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # logic module
    logic = LogicLossModule(
        deep_dfa=deep_dfa,
        adapter=adapter,
        mode=getattr(args, "logic_mode", "global"),
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

            # optional rollout eval (CB only)
            if getattr(args, "rollout_eval", False) and getattr(args, "benchmark", "") == "cb":
                env = dataset.env
                rets, traces = rollout_eval_cb(
                    model=model,
                    env=env,
                    adapter=adapter,
                    device=device,
                    n_episodes=getattr(args, "rollout_episodes", 50),
                    max_steps=getattr(args, "rollout_max_steps", None),
                    model_kind=getattr(args, "model", "tt"),
                    v_bins=getattr(args, "v_bins", 1),
                    dt_target_vtoken=getattr(args, "dt_target_vtoken", None),
                )

                # compute DFA satisfaction for each rollout trace
                sat_list = []
                for flat_tok in traces:
                    preds = flat_tok

                    if isinstance(raw_dfa, (list, tuple)):
                        sats = [adapter.batch_check_dfa_sat(preds, d, device=str(device)) for d in raw_dfa]
                        sat = torch.stack(sats, dim=0).min(dim=0).values
                    else:
                        sat = adapter.batch_check_dfa_sat(preds, raw_dfa, device=str(device))
                    sat_list.append(float(sat.item()))

                val_metrics["rollout_return_mean"] = float(np.mean(rets))
                val_metrics["rollout_return_std"] = float(np.std(rets))
                val_metrics["rollout_satisfaction_rate"] = float(np.mean(sat_list))


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
