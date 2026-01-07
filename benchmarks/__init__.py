# benchmarks/__init__.py
from benchmarks.cb_wrapper import ColourBombBenchmark
from benchmarks.nrm_nav_wrapper import NrmNavBenchmark

def get_benchmark(name: str):
    name = name.lower()
    if name in ["cb", "colour_bomb", "color_bomb"]:
        return ColourBombBenchmark()
    if name in ["nrm", "nrm_nav", "nav"]:
        return NrmNavBenchmark()
    raise ValueError(f"Unknown benchmark '{name}'. Use one of: cb, nrm_nav")
