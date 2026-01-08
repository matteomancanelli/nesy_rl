# eval.py
import argparse
import os
import torch

from benchmarks import get_benchmark
from run import build_model, eval_model
from helpers.checkpointing import load_checkpoint
from helpers.io import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, required=True, choices=["cb", "nrm_nav"])
    p.add_argument("--run_dir", type=str, required=True, help="e.g., runs/cb_seed0/gll_alpha0.4")
    p.add_argument("--ltl_formula", type=str, default=None)
    p.add_argument("--ltl_formulas", nargs="*", default=None)
    p.add_argument("--dfa_mode", type=str, default="single", choices=["single", "multi", "product"])
    p.add_argument("--use_safe_dfa", action="store_true")
    p.add_argument("--constraint_dims", nargs="*", type=int, default=None)

    # minimal dataset/model args required by benchmark + model
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block_size", type=int, default=200)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--embd_pdrop", type=float, default=0.1)
    p.add_argument("--resid_pdrop", type=float, default=0.1)
    p.add_argument("--attn_pdrop", type=float, default=0.1)
    p.add_argument("--action_weight", type=float, default=1.0)
    p.add_argument("--reward_weight", type=float, default=1.0)
    p.add_argument("--value_weight", type=float, default=1.0)

    p.add_argument("--num_episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--stochastic", action="store_true")

    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--save_traces", action="store_true")

    args = p.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bench = get_benchmark(args.benchmark)
    assets = bench.build_assets(args)

    # build model with correct vocab convention
    model, _cfg = build_model(args, assets.dataset, vocab_size=assets.adapter.num_token_ids, device=device)

    ckpt_path = os.path.join(args.run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    epoch, _extra = load_checkpoint(ckpt_path, model, optimizer=None, map_location=str(device), restore_rng=False)
    print(f"Loaded checkpoint epoch={epoch} from {ckpt_path}")

    from torch.utils.data import DataLoader
    loader = DataLoader(assets.dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    metrics = eval_model(
        model, assets.adapter, assets.raw_dfa, loader, device,
        save_traces_path=os.path.join(args.run_dir, f"eval_preds_epoch{epoch}.npz") if args.save_traces else None
    )
    save_json(os.path.join(args.run_dir, "eval_metrics.json"), metrics)
    print("Eval:", metrics)


if __name__ == "__main__":
    main()
