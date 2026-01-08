from dataclasses import dataclass
from typing import Any, List, Optional

from dfa_adapter import TTDFAAdapter
from benchmarks.cb_dataset import CBSequenceDataset


@dataclass
class BenchmarkAssets:
    dataset: Any
    adapter: TTDFAAdapter
    deep_dfa: Any
    raw_dfa: Any
    meta: dict


class ColourBombBenchmark:
    name = "cb"

    def _avoid_bombs_formula(self, env) -> str:
        bomb_ids = [s for s in range(env.observation_space.n) if env.is_bomb_state(s)]
        if not bomb_ids:
            raise ValueError("No bomb states found; cannot build avoid-bombs constraint.")
        inner = " | ".join([f"s0_bin{b}" for b in bomb_ids])
        return f"G(!({inner}))"

    def make_dataset(self, args):
        # cache path (optional)
        dataset_path = None
        if getattr(args, "dataset_cache_dir", None):
            #dataset_path = f"{args.dataset_cache_dir}/cb_ep{args.num_episodes}_ms{args.max_steps}_seed{args.seed}.npz"
            model = getattr(args, "model", "tt")
            v_bins = getattr(args, "v_bins", 1)
            stoch = int(bool(getattr(args, "stochastic", False)))
            disc = float(getattr(args, "discount", 0.99))

            dataset_path = (
                f"{args.dataset_cache_dir}/cb_{model}"
                f"_vb{v_bins}"
                f"_ep{args.num_episodes}"
                f"_ms{args.max_steps}"
                f"_disc{disc:.4f}"
                f"_st{stoch}"
                f"_seed{args.seed}.npz"
            )


        return CBSequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=args.block_size,
            discount=args.discount,
            stochastic=args.stochastic,
            seed=args.seed,
            dataset_path=dataset_path,
            v_bins=getattr(args, "v_bins", 1),
            model=getattr(args, "model", "tt")
        )

    def make_adapter_and_dfa(self, args, dataset):
        env = dataset.env

        obs_bins = env.observation_space.n
        act_bins = env.action_space.n
        rew_bins = 1
        val_bins = getattr(args, "v_bins", 1)


        adapter = TTDFAAdapter(
            observation_dim=dataset.observation_dim,
            action_dim=dataset.action_dim,
            num_bins=[obs_bins, act_bins, rew_bins, val_bins],
            include_reward=True,
            include_value=True,
            constraint_dims=args.constraint_dims,
            abstraction_fn=None,
            use_stop_token=True,
        )

        # choose formulas
        formulas: List[str] = []
        if args.ltl_formulas:
            formulas = list(args.ltl_formulas)
        elif args.ltl_formula:
            formulas = [args.ltl_formula]
        else:
            # default for CB toys: avoid all bombs
            formulas = [self._avoid_bombs_formula(env)]

        dfas = [adapter.create_dfa_from_ltl(f, f"cb_constraint_{i}") for i, f in enumerate(formulas)]

        # single only for toys (keep it simple)
        raw_dfa = dfas[0] if len(dfas) == 1 else dfas
        deep_dfa = raw_dfa.return_deep_dfa() if not isinstance(raw_dfa, list) else [d.return_deep_dfa() for d in raw_dfa]

        meta = {
            "default_formula_used": (args.ltl_formula is None and not args.ltl_formulas),
            "formulas": formulas,
        }
        return adapter, deep_dfa, raw_dfa, meta

    def build_assets(self, args) -> BenchmarkAssets:
        dataset = self.make_dataset(args)
        adapter, deep_dfa, raw_dfa, meta = self.make_adapter_and_dfa(args, dataset)
        return BenchmarkAssets(dataset=dataset, adapter=adapter, deep_dfa=deep_dfa, raw_dfa=raw_dfa, meta=meta)
