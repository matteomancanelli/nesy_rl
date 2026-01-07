# train.py
import argparse
import os
import torch

from benchmarks import get_benchmark
from run import train_one_run


def get_arg_parser():
    p = argparse.ArgumentParser()

    # benchmark + constraints
    p.add_argument("--benchmark", type=str, default="cb", choices=["cb", "nrm_nav"])
    p.add_argument("--ltl_formula", type=str, default=None)
    p.add_argument("--ltl_formulas", nargs="*", default=None)
    p.add_argument("--dfa_mode", type=str, default="single", choices=["single", "multi", "product"])
    p.add_argument("--use_safe_dfa", action="store_true")
    p.add_argument("--constraint_dims", nargs="*", type=int, default=None)

    # training
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # TT config
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

    # dataset/env generation knobs used by your toy datasets
    p.add_argument("--num_episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--stochastic", action="store_true")

    # logic loss
    p.add_argument("--alpha", type=float, default=0.4, help="Logic weight for TT+GLL run")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--logic_eps", type=float, default=1e-10)
    p.add_argument("--no_logic_clamp", action="store_true")

    # eval + outputs
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_max_batches", type=int, default=None)
    p.add_argument("--out_root", type=str, default="runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_traces", action="store_true")
    p.add_argument("--dataset_cache_dir", type=str, default="data_cache")

    return p


def main():
    args = get_arg_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bench = get_benchmark(args.benchmark)
    assets = bench.build_assets(args)

    tag = args.run_name or f"{args.benchmark}_seed{args.seed}"
    base = os.path.join(args.out_root, tag)

    # A) vanilla TT
    train_one_run(args, assets, out_dir=os.path.join(base, "vanilla"), alpha=0.0, device=device)

    # B) TT + global logic loss
    train_one_run(args, assets, out_dir=os.path.join(base, f"gll_alpha_{args.alpha}"), alpha=args.alpha, device=device)


if __name__ == "__main__":
    main()
