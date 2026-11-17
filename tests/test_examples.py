import importlib.util
import sys
from pathlib import Path


def run_example(name: str) -> str:
    path = Path(__file__).parent.parent / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return "\n".join(module.main())


def test_quickstart_tells_the_whole_story():
    out = run_example("quickstart")
    assert "Bo's agent recalls:  []" in out
    assert "Today: ['Ana works at Hooli']. As of June: ['Ana works at Initech']" in out
    assert "shared by ana" in out
    assert "After revoking, Bo sees []" in out
    assert "(graph rank 1)" in out
    assert "Supplier card is [CARD], contact [EMAIL]" in out
    assert "Injected instruction surfaces in recall: False" in out
    assert "<recalled_memory>" in out
