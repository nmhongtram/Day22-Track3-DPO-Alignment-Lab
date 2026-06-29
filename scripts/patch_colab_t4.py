#!/usr/bin/env python3
"""One-shot patch for colab/Lab22_DPO_T4.ipynb — apply plan fixes."""
from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_PATH = REPO / "colab" / "Lab22_DPO_T4.ipynb"
REFLECTION = REPO / "submission" / "REFLECTION.md"


def cell_id() -> str:
    return uuid.uuid4().hex[:8]


def md_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in lines],
    }


def code_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in lines],
    }


def join_source(src: list[str]) -> str:
    return "".join(src)


def set_source(cell: dict, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n") + "\n"


def insert_after(cells: list, marker: str, new_cells: list[dict], in_markdown: bool = False) -> bool:
    for i, c in enumerate(cells):
        src = join_source(c.get("source", []))
        if marker in src and (not in_markdown or c["cell_type"] == "markdown"):
            cells[i + 1 : i + 1] = new_cells
            return True
        if marker in src and not in_markdown and c["cell_type"] == "code":
            cells[i + 1 : i + 1] = new_cells
            return True
    return False


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- Cell 0: intro markdown ---
    intro = join_source(cells[0]["source"])
    intro = intro.replace(
        "> **Before running:** Runtime → Change runtime type → T4 GPU (free).\n"
        "> Verify with `nvidia-smi` cell below.",
        "> **Before running:** Runtime → Change runtime type → **T4 GPU** (free).\n"
        "> **Two-pass workflow:** Pass 1 installs deps and restarts runtime automatically.\n"
        "> Pass 2 (Run All again) runs NB1–NB6 (~60–90 min full / ~30 min core NB1–NB4).\n"
        "> Verify GPU with the `nvidia-smi` cell in Section A.",
    )
    set_source(cells[0], intro.rstrip("\n"))

    # --- Cell 1: Section A header ---
    set_source(
        cells[1],
        "## A. Colab setup — install deps + set tier\n"
        "(Skip if running from a cloned lab repo with deps already installed.)",
    )

    # --- Cell 2: tier + env ---
    set_source(
        cells[2],
        """# Set tier + env vars — every downstream cell reads these.
import os
from pathlib import Path

os.environ["COMPUTE_TIER"] = "T4"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print(f"COMPUTE_TIER={os.environ['COMPUTE_TIER']}")""",
    )

    # --- Cell 3: pip install with marker + transformers ---
    set_source(
        cells[3],
        """# Install deps (~3-5 min). Skipped on pass 2 after runtime restart.
from pathlib import Path

DEPS_MARKER = Path("/content/.lab22_deps_ok")

if not DEPS_MARKER.exists():
    import subprocess
    import sys

    pkgs = [
        "unsloth>=2025.10,<2026.5",
        "transformers>=4.46,<5.0",
        "trl>=0.12,<0.20",
        "peft>=0.13,<1.0",
        "bitsandbytes>=0.44,<1.0",
        "datasets>=3.1,<4.0",
        "accelerate>=1.1,<2.0",
        "matplotlib>=3.9,<4.0",
        "pandas>=2.2,<3.0",
        "pyarrow>=17,<22",
        "openai>=1.55,<2.0",
        "anthropic>=0.40,<1.0",
        "lm-eval[ifeval,math]>=0.4.5,<1.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    # llama-cpp with CUDA when possible (smoke test in NB5)
    subprocess.call(
        'CMAKE_ARGS="-DGGML_CUDA=on" pip install -q --force-reinstall "llama-cpp-python>=0.3,<1.0"',
        shell=True,
    )
    DEPS_MARKER.write_text("ok")
    print("Deps installed. Restarting runtime — click Run All again when Colab reloads.")
    import os as _os
    _os.kill(_os.getpid(), 9)
else:
    print("Deps already installed (pass 2) — continuing to NB1.")""",
    )

    # --- Insert nvidia-smi before GPU probe (cell 4) ---
    nvidia_cell = code_cell(["!nvidia-smi"])
    cells.insert(4, nvidia_cell)

    # --- WORK dir cell (now index 6 after insert) ---
    work_cell_idx = 6
    set_source(
        cells[work_cell_idx],
        '''# Colab workspace — mirrors repo layout under /content/lab22
from pathlib import Path
import os

WORK = Path("/content/lab22")
WORK.mkdir(exist_ok=True)
for sub in [
    "notebooks",
    "data/pref",
    "data/eval",
    "adapters/sft-mini",
    "adapters/dpo",
    "adapters/sft-dpo-stacked",
    "adapters/merged-fp16",
    "gguf",
    "submission/screenshots",
]:
    (WORK / sub).mkdir(parents=True, exist_ok=True)

reflection = WORK / "submission" / "REFLECTION.md"
if not reflection.exists():
    reflection.write_text(
        "# Reflection — Lab 22\\n\\n"
        "**Tên:** _<Họ Tên>_\\n"
        "**Tier:** T4\\n\\n"
        "Fill all sections before VinUni LMS submit. "
        "See lab repo submission/REFLECTION.md for full template.\\n",
        encoding="utf-8",
    )
    print(f"Created {reflection} — fill before LMS submit.")

os.chdir(WORK / "notebooks")
print(f"Working dir: {Path.cwd()}")''',
    )

    # --- Stitch header ---
    for c in cells:
        if "## Stages 1-5 stitched below" in join_source(c["source"]):
            set_source(
                c,
                "---\n"
                "## Stages 1–6 stitched below\n"
                "Run cells in order. **If OOM at NB3:** run the VRAM cleanup cell after NB1,\n"
                "or Runtime → Restart → rerun from NB2 (adapters on disk are kept).\n"
                "Set `os.environ['PREF_SLICE']='1000'` before NB2 if still OOM.\n"
                "---",
            )
            break

    # --- Bulk text replacements in all code/markdown cells ---
    replacements = [
        # NB2 PREF_SLICE
        (
            '    PREF_SLICE = 2000\n',
            '    PREF_SLICE = int(os.environ.get("PREF_SLICE", "2000"))  # set 1000 if OOM\n',
        ),
        (
            '    PREF_SLICE = 5000\n',
            '    PREF_SLICE = int(os.environ.get("PREF_SLICE", "5000"))\n',
        ),
        # NB3 markdown wrong VRAM text
        (
            "## 1. Load policy + reference (the VRAM-doubling part)\n",
            "## 1. Load policy + reference (the VRAM story)\n",
        ),
        (
            "The reference is the SFT model at step 0; we load it twice. Unsloth's 4-bit base\n"
            "is shared across copies — only the LoRA adapter differs.",
            "**Critical:** With PEFT we do **not** load a second model — TRL toggles the LoRA\n"
            "adapter *off* for the reference forward pass on the same 4-bit base. Extra VRAM\n"
            "vs SFT comes from two forward passes + chosen AND rejected in batch.",
        ),
        # NB3 metrics
        (
            '    "end_chosen_reward": float(last_chosen) if chosen_col else None,\n'
            '    "end_rejected_reward": float(last_rejected) if rejected_col else None,\n'
            '    "end_reward_gap": float(last_gap) if chosen_col and rejected_col else None,\n',
            '    "end_chosen_reward": end_chosen,\n'
            '    "end_rejected_reward": end_rejected,\n'
            '    "end_reward_gap": end_gap,\n',
        ),
        # NB4 generate
        (
            "def generate_with_adapter(adapter_path: Path, prompts: list[dict], max_new_tokens: int = 256):",
            "def generate_with_adapter(adapter_path: Path, prompts: list[dict], max_new_tokens: int = 256, stack_sft: bool = False):",
        ),
        (
            '    model = PeftModel.from_pretrained(model, str(adapter_path))\n'
            '    FastLanguageModel.for_inference(model)',
            '    if stack_sft:\n'
            '        model = PeftModel.from_pretrained(model, str(SFT_PATH))\n'
            '    model = PeftModel.from_pretrained(model, str(adapter_path))\n'
            '    FastLanguageModel.for_inference(model)',
        ),
        (
            'dpo_outputs = generate_with_adapter(DPO_PATH, EVAL_PROMPTS)',
            'dpo_outputs = generate_with_adapter(DPO_PATH, EVAL_PROMPTS, stack_sft=True)',
        ),
        # NB5 load DPO
        (
            'model = PeftModel.from_pretrained(model, str(SFT_PATH))\n'
            'print(f"Loaded SFT-mini adapter from {SFT_PATH}")',
            'model = PeftModel.from_pretrained(model, str(SFT_PATH))\n'
            'print(f"Loaded SFT-mini adapter from {SFT_PATH}")\n'
            '\n'
            'model = PeftModel.from_pretrained(model, str(DPO_PATH))\n'
            'print(f"Loaded DPO adapter from {DPO_PATH}")',
        ),
        # NB5 llama-cpp
        (
            "# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan\n"
            "llm = Llama(\n"
            '    model_path=str(gguf_path),\n'
            "    n_ctx=MAX_LEN,\n"
            "    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile\n"
            "    verbose=False,\n"
            ")\n"
            'print("Loaded.")',
            "try:\n"
            "    from llama_cpp import llama_cpp\n"
            "    has_cuda = getattr(llama_cpp, 'llama_supports_gpu_offload', lambda: False)()\n"
            "except Exception:\n"
            "    has_cuda = False\n"
            "n_gpu = -1 if has_cuda else 0\n"
            "if not has_cuda:\n"
            '    print("WARN: llama-cpp CPU build — n_gpu_layers=0 (slower smoke test).")\n'
            "\n"
            "llm = Llama(\n"
            '    model_path=str(gguf_path),\n'
            "    n_ctx=MAX_LEN,\n"
            "    n_gpu_layers=n_gpu,\n"
            "    verbose=False,\n"
            ")\n"
            'print("Loaded.")',
        ),
        # NB5/NB6 titles optional
        (
            "# NB5 — Merge + Deploy + GGUF\n",
            "# NB5 — Merge + Deploy + GGUF  (OPTIONAL / BONUS)\n\n"
            "> **Optional.** Core lab = NB1–NB4. Skip if short on time.\n",
        ),
        (
            "# NB6 — LLM Benchmark: SFT-only vs SFT+DPO\n",
            "# NB6 — LLM Benchmark: SFT-only vs SFT+DPO  (OPTIONAL / BONUS)\n\n"
            "> **Optional.** ~30+ min on T4. Skip if short on time.\n",
        ),
        # Submission callouts
        (
            "1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.",
            "1. **Checklist:** `adapters/sft-mini/`, `adapters/dpo/`, `data/pref/train.parquet`, screenshots in `submission/screenshots/`, filled `submission/REFLECTION.md`.",
        ),
        # NB6 generate stack
        (
            "def generate_with_adapter(adapter_path, prompts, max_new_tokens=256):",
            "def generate_with_adapter(adapter_path, prompts, max_new_tokens=256, stack_sft=False):",
        ),
        (
            '    model = PeftModel.from_pretrained(model, str(adapter_path))\n'
            '    FastLanguageModel.for_inference(model)\n'
            '\n'
            '    outputs = []',
            '    if stack_sft:\n'
            '        model = PeftModel.from_pretrained(model, str(SFT_PATH))\n'
            '    model = PeftModel.from_pretrained(model, str(adapter_path))\n'
            '    FastLanguageModel.for_inference(model)\n'
            '\n'
            '    outputs = []',
        ),
        (
            '    dpo_outputs = generate_with_adapter(DPO_PATH, alpaca_prompts)',
            '    dpo_outputs = generate_with_adapter(DPO_PATH, alpaca_prompts, stack_sft=True)',
        ),
        # NB6 lm-eval DPO paths
        (
            'dpo_ifeval = run_lm_eval(DPO_PATH, "ifeval", LIMIT_IFEVAL, num_fewshot=0, label="dpo")',
            'dpo_ifeval = run_lm_eval(STACKED_PATH, "ifeval", LIMIT_IFEVAL, num_fewshot=0, label="dpo", use_peft=False)',
        ),
        (
            'dpo_gsm8k = run_lm_eval(DPO_PATH, "gsm8k", LIMIT_GSM8K, num_fewshot=8, label="dpo")',
            'dpo_gsm8k = run_lm_eval(STACKED_PATH, "gsm8k", LIMIT_GSM8K, num_fewshot=8, label="dpo", use_peft=False)',
        ),
        (
            'dpo_mmlu = run_lm_eval(DPO_PATH, "mmlu", LIMIT_MMLU, num_fewshot=5, label="dpo")',
            'dpo_mmlu = run_lm_eval(STACKED_PATH, "mmlu", LIMIT_MMLU, num_fewshot=5, label="dpo", use_peft=False)',
        ),
    ]

    metrics_prefix = (
        "# Save the headline metrics for verify.py + REFLECTION\n"
        "import json\n"
        "\n"
        "end_chosen = float(logs[chosen_col].iloc[-5:].mean()) if chosen_col and len(logs) >= 5 else None\n"
        "end_rejected = float(logs[rejected_col].iloc[-5:].mean()) if rejected_col and len(logs) >= 5 else None\n"
        "end_gap = (end_chosen - end_rejected) if end_chosen is not None and end_rejected is not None else None\n"
        "\n"
        "metrics = {"
    )

    for c in cells:
        src = join_source(c["source"])
        for old, new in replacements:
            src = src.replace(old, new)
        if (
            '"end_chosen_reward": end_chosen' in src
            and "end_chosen = float(logs" not in src
        ):
            src = src.replace(
                "# Save the headline metrics for verify.py + REFLECTION\nimport json\n\nmetrics = {",
                metrics_prefix,
            )
        set_source(c, src.rstrip("\n"))

    # --- NB6: replace run_lm_eval function + add ensure_stacked if missing ---
    for i, c in enumerate(cells):
        src = join_source(c["source"])
        if c["cell_type"] == "code" and "def run_lm_eval(adapter_path" in src and "ensure_stacked_model" not in src:
            set_source(
                c,
                '''import subprocess


def ensure_stacked_model():
    """Export SFT+DPO merged weights for lm-eval (single peft= path is insufficient)."""
    stacked_dir = REPO_ROOT / "adapters" / "sft-dpo-stacked"
    if (stacked_dir / "config.json").exists():
        return stacked_dir

    from peft import PeftModel
    from unsloth import FastLanguageModel

    base = "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    max_len = 512 if COMPUTE_TIER == "T4" else 1024

    print(f"Building stacked SFT+DPO export at {stacked_dir} (one-time, ~2 min)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, str(SFT_PATH))
    model = PeftModel.from_pretrained(model, str(DPO_PATH))
    model = model.merge_and_unload()
    stacked_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(stacked_dir))
    tokenizer.save_pretrained(str(stacked_dir))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return stacked_dir


def run_lm_eval(model_path, tasks, limit, num_fewshot, label, use_peft=True):
    """Run lm-eval-harness; use use_peft=False for merged stacked model dir."""
    base = "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    out_dir = EVAL_OUT / f"lm-{label}-{tasks}"
    if use_peft:
        model_args = f"pretrained={base},peft={model_path},load_in_4bit=True"
    else:
        model_args = f"pretrained={model_path},load_in_4bit=True"
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--batch_size", str(BATCH_SIZE),
        "--device", "cuda:0",
        "--output_path", str(out_dir),
    ]
    print(f"\\n{'=' * 60}\\nRunning lm-eval [{label}]: {tasks}\\n{'=' * 60}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)

    out_files = sorted(out_dir.glob("**/results*.json"))
    if not out_files:
        print("WARN: lm-eval didn't write results JSON. STDOUT tail:")
        print(proc.stdout[-1000:] if proc.stdout else "(empty stdout)")
        if proc.stderr:
            print("STDERR tail:")
            print(proc.stderr[-1000:])
        return {"error": "no_results"}
    return json.loads(out_files[-1].read_text())["results"]''',
            )
            # Insert stacked prep cells after this one
            if i + 1 < len(cells) and "STACKED_PATH" not in join_source(cells[i + 1]["source"]):
                cells.insert(i + 1, md_cell(["## 1b. Prep stacked SFT+DPO export for lm-eval"]))
                cells.insert(
                    i + 2,
                    code_cell(
                        [
                            "STACKED_PATH = ensure_stacked_model()",
                            'print(f"Stacked model ready: {STACKED_PATH}")',
                        ]
                    ),
                )
            break

    # --- VRAM bridge after NB1 sanity generation ---
    nb1_bridge_md = md_cell(["### Colab VRAM cleanup — run before NB3"])
    nb1_bridge_code = code_cell(
        [
            "# Free GPU after SFT — required for all-in-one Colab Run All",
            "for _var in ('model', 'trainer', 'train_result', 'ds', 'ds_formatted'):",
            "    if _var in globals():",
            "        del globals()[_var]",
            "import gc",
            "gc.collect()",
            "torch.cuda.empty_cache()",
            'print(f"VRAM after NB1 cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")',
        ]
    )
    if not insert_after(cells, "print(f\"SFT-mini response", [nb1_bridge_md, nb1_bridge_code]):
        insert_after(cells, "SFT-mini response", [nb1_bridge_md, nb1_bridge_code])

    # --- VRAM bridge after NB3 save ---
    nb3_bridge_md = md_cell(["### Colab VRAM cleanup — run before NB4"])
    nb3_bridge_code = code_cell(
        [
            "for _var in ('model', 'trainer', 'train_result', 'pref_ds', 'logs'):",
            "    if _var in globals():",
            "        del globals()[_var]",
            "gc.collect()",
            "torch.cuda.empty_cache()",
            'print(f"VRAM after NB3 cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")',
        ]
    )
    if not insert_after(cells, 'print(f"Wrote metrics to {DPO_OUT / \'dpo_metrics.json\'}")', [nb3_bridge_md, nb3_bridge_code]):
        insert_after(cells, "dpo_metrics.json", [nb3_bridge_md, nb3_bridge_code])

    # --- VRAM bridge before NB5 ---
    nb5_bridge_md = md_cell(["### Colab VRAM cleanup — run before NB5 (optional)"])
    nb5_bridge_code = code_cell(
        [
            "import gc",
            "gc.collect()",
            "torch.cuda.empty_cache()",
            'print(f"VRAM before NB5: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")',
        ]
    )
    if not insert_after(cells, "# ⏵ Stage from `notebooks/05_merge_deploy_gguf.py`", [nb5_bridge_md, nb5_bridge_code], in_markdown=True):
        pass

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
