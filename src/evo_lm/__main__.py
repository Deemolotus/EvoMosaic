from evo_lm.cli import prepare

if __name__ == "__main__":
    # default module entry: show help via subcommands by env or argv
    import sys

    cmds = {
        "prepare": "evo_lm.cli:prepare",
        "evolve": "evo_lm.cli:evolve",
        "train": "evo_lm.cli:train",
        "chat": "evo_lm.cli:chat",
        "export-latex": "evo_lm.cli:export_latex",
    }
    if len(sys.argv) > 1 and sys.argv[1] in cmds:
        name = sys.argv.pop(1)
        mod, fn = cmds[name].split(":")
        import importlib

        getattr(importlib.import_module(mod), fn)()
    else:
        print("Usage: python -m evo_lm <prepare|evolve|train|chat|export-latex> ...")
        prepare.__doc__ and None
        sys.exit(1)
