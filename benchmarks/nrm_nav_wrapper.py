# benchmarks/nrm_nav.py
from dataclasses import dataclass
from typing import Any, List

from dfa_adapter import TTDFAAdapter

from benchmarks.nrm_nav_dataset import NRMSafetySequenceDataset


@dataclass
class BenchmarkAssets:
    dataset: Any
    adapter: TTDFAAdapter
    deep_dfa: Any
    raw_dfa: Any


class NrmNavBenchmark:
    name = "nrm_nav"

    def make_dataset(self, args):
        return NRMSafetySequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=args.block_size,
            discount=args.discount,
            stochastic=args.stochastic,
            seed=args.seed,
            grid=None,
        )

    def make_adapter_and_dfa(self, args, dataset):
        env = dataset.env

        obs_bins = env.observation_space.n
        act_bins = env.action_space.n
        rew_bins = 1
        val_bins = 1
        num_bins_per_dim = [obs_bins, act_bins, rew_bins, val_bins]

        adapter = TTDFAAdapter(
            observation_dim=dataset.observation_dim,
            action_dim=dataset.action_dim,
            num_bins=num_bins_per_dim,
            include_reward=True,
            include_value=True,
            constraint_dims=args.constraint_dims,
            abstraction_fn=None,
            use_stop_token=True,
        )

        formulas: List[str] = []
        if args.ltl_formulas is not None and len(args.ltl_formulas) > 0:
            formulas = args.ltl_formulas
        elif args.ltl_formula is not None:
            formulas = [args.ltl_formula]
        else:
            raise ValueError("Provide --ltl_formula or --ltl_formulas")

        dfas = [
            adapter.create_dfa_from_ltl(f, f"nrm_constraint_{i}", use_safe_dfa=args.use_safe_dfa)
            for i, f in enumerate(formulas)
        ]

        if len(dfas) == 1 or args.dfa_mode == "single":
            raw_dfa = dfas[0]
            deep_dfa = raw_dfa.return_deep_dfa()
        elif args.dfa_mode == "multi":
            raw_dfa = dfas
            deep_dfa = [d.return_deep_dfa() for d in dfas]
        else:
            raise ValueError("For now, nrm_nav supports dfa_mode in {single, multi} (add product if needed).")

        return adapter, deep_dfa, raw_dfa

    def build_assets(self, args) -> BenchmarkAssets:
        dataset = self.make_dataset(args)
        adapter, deep_dfa, raw_dfa = self.make_adapter_and_dfa(args, dataset)
        return BenchmarkAssets(dataset=dataset, adapter=adapter, deep_dfa=deep_dfa, raw_dfa=raw_dfa)
