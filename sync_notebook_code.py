#!/usr/bin/env python3
"""Regenerate PersonaPlex_Student_Training_fixed.ipynb's %%writefile cells from the files in distill/.

The training notebook writes its own copies of several distill/ modules into REPO_DIR at run time ("## 4.
Install the updated training code into the repo"), so an edit made only on disk is silently overwritten when
the notebook runs. Before this script existed the two had already diverged: the notebook's train.py and
student_model.py were newer than the checkout's.

Run this after editing any file listed in CELL_TO_PATH. It also ADDS a %%writefile cell for a mapped file the
notebook does not write yet, so new modules (e.g. distill/data/text_dataset.py) reach the pod too.
"""
import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("PersonaPlex_Student_Training_fixed.ipynb")

# published path inside REPO_DIR  ->  source file in this checkout
PATHS = [
    "distill/train.py",
    "distill/checkpoint.py",
    "distill/export.py",
    "distill/data/dataset.py",
    "distill/data/text_dataset.py",
    "distill/gqa_attention.py",
    "distill/student_model.py",
    "distill/schedule.py",
    "distill/losses.py",
    "distill/eval.py",
    "distill/init_from_teacher.py",
    # The pod's distill/ otherwise comes from the GitHub clone, which has no student_ppx_m.yaml -- without
    # these two lines `STUDENT_CONFIG = "student_ppx_m"` dies in load_student_config with FileNotFoundError.
    "distill/configs/student_ppx_s.yaml",
    "distill/configs/student_ppx_m.yaml",
    # tools/ is not in the GitHub clone either, and both of these are needed ON the pod:
    # build_text_corpus.py rebuilds /workspace/ppx_text_corpus, and export_checkpoint.py is the only way
    # to listen to a mid-run checkpoint (the trainer exports only when it reaches --total-steps).
    "tools/build_text_corpus.py",
    "tools/verify_text_corpus.py",
    "tools/export_checkpoint.py",
]


def magic_for(path: str) -> str:
    return "%%writefile {REPO_DIR}/" + path


MKDIR_CELL = [
    "import os\n",
    "# %%writefile does not create folders, so every directory the cells below write into must exist first.\n",
    'for _d in ("distill/data", "distill/configs", "tools"):\n',
    "    os.makedirs(os.path.join(REPO_DIR, _d), exist_ok=True)\n",
    'print("writing updated training code into", REPO_DIR)\n',
]


def ensure_dirs_cell(nb) -> bool:
    """Make the notebook's mkdir cell create every directory the %%writefile cells target."""
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "os.makedirs(os.path.join(REPO_DIR" in src and '"tools"' not in src:
            c["source"] = MKDIR_CELL
            return True
    return False


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if ensure_dirs_cell(nb):
        print("  UPDATE mkdir cell (now creates distill/configs too)")
    by_path = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        first = "".join(c["source"]).split("\n", 1)[0].strip()
        if first.startswith("%%writefile {REPO_DIR}/"):
            by_path[first[len("%%writefile {REPO_DIR}/"):]] = i

    anchor = max(by_path.values()) if by_path else None
    updated, added, unchanged = [], [], []
    for path in PATHS:
        src = Path(path)
        if not src.exists():
            print(f"  SKIP   {path} (not in this checkout)")
            continue
        body = src.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
        new_source = (magic_for(path) + "\n" + body + "\n").splitlines(keepends=True)
        if path in by_path:
            idx = by_path[path]
            if "".join(nb["cells"][idx]["source"]) == "".join(new_source):
                unchanged.append(path)
            else:
                nb["cells"][idx]["source"] = new_source
                nb["cells"][idx]["outputs"] = []
                nb["cells"][idx]["execution_count"] = None
                updated.append(path)
        else:
            cell = {"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [], "source": new_source}
            anchor = (anchor + 1) if anchor is not None else len(nb["cells"])
            nb["cells"].insert(anchor, cell)
            added.append(path)
            by_path[path] = anchor

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for tag, items in (("UPDATE", updated), ("ADD", added), ("same", unchanged)):
        for p in items:
            print(f"  {tag:7s}{p}")
    print(f"\n{NOTEBOOK.name}: {len(updated)} updated, {len(added)} added, {len(unchanged)} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
