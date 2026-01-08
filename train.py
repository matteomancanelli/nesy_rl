# train.py
import argparse
import os
import torch
import copy

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

    # dataset/env generation knobs used by toy datasets
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

    p.add_argument("--models", nargs="*", default=["tt"], choices=["tt", "dt"],
        help="Which models to run. Default: tt. Use: --models tt dt")
    p.add_argument("--logic_mode", type=str, default="global", choices=["global", "local"])
    p.add_argument("--v_bins", type=int, default=1, help="Number of bins for v. For DT, set e.g. 50 or 100.")

    p.add_argument("--rollout_eval", action="store_true",
               help="If set, run closed-loop env rollouts during eval (CB only for now).")
    p.add_argument("--rollout_episodes", type=int, default=50)
    p.add_argument("--rollout_max_steps", type=int, default=None,
                help="Override env max_steps for rollout eval if set.")
    p.add_argument("--dt_target_vtoken", type=int, default=None,
                help="For DT rollouts: fixed v-token to condition on (default: v_bins-1).")

    return p

def main():
    args = get_arg_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_name in args.models:
        args_m = copy.deepcopy(args)
        args_m.model = model_name

        if model_name == "dt":
            if args_m.v_bins <= 1:
                args_m.v_bins = 50  # default RTG bins
            args_m.reward_weight = 0.0
            args_m.value_weight = 0.0
            args_m.action_weight = 1.0

        bench = get_benchmark(args_m.benchmark)
        assets = bench.build_assets(args_m)

        tag = args_m.run_name or f"{args_m.benchmark}_{model_name}_seed{args_m.seed}"
        base = os.path.join(args_m.out_root, tag)

        # A) vanilla
        train_one_run(args_m, assets, out_dir=os.path.join(base, "vanilla"),
                      alpha=0.0, device=device)

        # B) logic-regularized
        train_one_run(args_m, assets, out_dir=os.path.join(base, f"{args_m.logic_mode}_alpha_{args_m.alpha}"),
                      alpha=args_m.alpha, device=device)


if __name__ == "__main__":
    main()
