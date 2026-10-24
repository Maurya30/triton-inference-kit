# Running the GPU validation on RunPod

One command on a rented A100 runs every test and benchmark, saves the results in `benchmarks/results/`, and rewrites the README and `docs/` tables from them. These are the exact steps, from a RunPod account with credit to the results pushed to GitHub. RunPod's console and prices were checked on 2026-10-23; if a label has moved, the choices below are what to look for.

## 1. Before you start (once)

- **RunPod credit.** RunPod only deploys a pod if the balance covers at least one hour of it, so keep $5 or more in the account. The run itself should cost under $2 (see [Runtime and cost](#runtime-and-cost)).
- **A GitHub token that can push to this repo.** On the Maurya30 account: *Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token*. Repository access: *Only select repositories → triton-inference-kit*. Permissions: *Contents: Read and write*. Expiration: 7 days. Copy it; GitHub shows it once.
- **Optional: a Hugging Face token** that can read `meta-llama/Llama-3.2-1B` (open the model page on huggingface.co and accept the license first). With it, the end-to-end test also runs the real 1B model; without it, that case is skipped and everything else still runs.

## 2. Deploy the pod

In the RunPod console (console.runpod.io): **Pods → + New → Pod** (older consoles: **Pods → Deploy**).

1. **GPU:** **A100 SXM** (80 GB), count 1, **Secure Cloud**. If none is available, **A100 PCIe** (80 GB) works too; the benchmarks detect the GPU and its peak bandwidth themselves. Prefer Secure Cloud: Community A100 capacity comes and goes, and has been seen on hosts whose CUDA driver is too old for the image below.
2. **Template:** the official **PyTorch** template (published by RunPod, also listed as "Runpod PyTorch"), version **2.13.0** (the default). In the CUDA dropdown pick **12.9.0 · Ubuntu 24.04**; the image shown under the template should read `runpod/pytorch:1.4.1-cu1290-torch2130-ubuntu2404`. That is PyTorch 2.13.0 built for CUDA 12.9 with the matching `nvcc` 12.9, Python 3.12 as `python`, and `git` and `tmux` preinstalled. If 12.9.0 is not offered, the default **12.8.1** tag of 2.13.0 also works: its `nvcc` is one minor version behind torch's CUDA, which the JIT build used here does not check. The run needs PyTorch 2.6 or newer (transformers 5.x refuses older torch) and `nvcc` (the CUDA C++ RMSNorm is compiled on first use); `scripts/run_a100.sh` records both in `env.txt` in its first step and stops there if either is missing. Do not pick the deprecated 2.1, 2.2 or 2.4 versions. If the pod fails to start with "OCI runtime create failed", the host's driver is too old for the image: use **Additional filters → CUDA Versions** to pick a newer host.
3. **Hugging Face token (optional):** in the template field, **Set overrides** (older consoles: **Edit Template**) → Environment Variables → add `HF_TOKEN` with your token as the value. To keep it out of the pod config, store it as a RunPod secret (console.runpod.io/user/secrets, e.g. named `huggingface_token`) and set the value to `{{ RUNPOD_SECRET_huggingface_token }}`. Either way, the second line `run_a100.sh` prints says `HF_TOKEN set` or `HF_TOKEN unset`; if it says unset by mistake, stop it (Ctrl-C), run `export HF_TOKEN=<your token>` and start it again.
4. **Disks:** keep the template defaults (50 GB container disk, 30 GB volume at `/workspace`). Model downloads go to `/workspace/.cache/huggingface`, about 5 GB at most.
5. **Pricing:** On-Demand. **Deploy**.

## 3. Open a terminal

When the pod shows *Running*: **Connect → Open Web Terminal** (press *Start* first if it is stopped). SSH works too; **Connect** shows the command once an SSH public key is added to your RunPod account.

The web terminal can drop on long jobs, so run inside `tmux` (preinstalled), which keeps going if the browser tab closes:

```bash
tmux new -s run
```

After a disconnect, open a terminal again and run `tmux attach -t run`.

## 4. Clone and run

```bash
cd /workspace
git clone https://github.com/Maurya30/triton-inference-kit.git
cd triton-inference-kit
bash scripts/run_a100.sh
```

The script prints a header for each step and stops at the first failure, naming the step. Everything it prints is also saved to `benchmarks/results/run_<timestamp>.log`. In order, it:

1. writes `nvidia-smi` and the torch/triton/CUDA/nvcc versions to `benchmarks/results/env.txt`;
2. runs `pip install -e ".[dev,bench]"`;
3. runs each `tests/test_*.py` on its own with `-v` (kernels first, the LLaMA end-to-end test last), saving a log and a JUnit XML per file in `benchmarks/results/tests/`;
4. runs every benchmark in float16 and bfloat16, writing `benchmarks/results/<gpu>_<date>.json` and charts;
5. runs `scripts/update_docs.py`, which fills the README and `docs/` tables and ticks the roadmap boxes whose tests passed and whose results exist.

A good run ends with `ALL STEPS PASSED`, the README results block, the total wall time and `git status`.

## 5. Check before committing

```bash
git status --short
git diff README.md docs/
```

If a step failed instead, the last lines name it and the log is in `benchmarks/results/`. Do not commit a partial run: fix the cause, then run `bash scripts/run_a100.sh` again. It starts over, and moves the failed attempt's untracked files to `/tmp/triton-inference-kit-earlier-attempts/` so they stay out of the commit. If you changed code to get a passing run, commit that change too: add its paths to the `git add` in step 6 (`env.txt` records the commit and lists uncommitted code changes).

## 6. Commit and push

```bash
git config user.name "Maurya Panchal"
git config user.email "m25panch@uwaterloo.ca"
git add benchmarks/results docs README.md
git commit -m "A100 run: test logs, benchmark results, docs"
git push origin HEAD:main
```

`git push` asks for a username and a password: enter `Maurya30`, then paste the fine-grained token from step 1 as the password. It is not saved on the pod.

## 7. Stop paying

On the Pods page, **Terminate** the pod (trash icon). *Stop* is not enough: a stopped pod keeps billing for its volume disk. Then delete the fine-grained token on GitHub.

## Runtime and cost

These are estimates, not measurements: nothing in this repo has run on a GPU yet. The script prints its real wall time at the end of the run (it is also in the run log), so this section can be corrected after the first run.

| Phase | Estimate | What dominates |
|---|---|---|
| Pod start | 2 to 10 min | pulling the roughly 16 GB image if the host has not cached it |
| `pip install` | 1 to 2 min | transformers and matplotlib; torch and Triton are already in the image |
| Tests | about 5 min | Triton JIT compiles and one `nvcc` build of the CUDA extension (plus a 2.5 GB download if `HF_TOKEN` is set) |
| Kernel benchmarks | about 10 min | 42 `torch.compile` builds (one per shape and dtype), not the timed runs |
| End-to-end benchmark | 5 to 10 min | a 2.2 GB TinyLlama download, then `generate` for stock and two patched variants in both dtypes |

So `bash scripts/run_a100.sh` should take roughly 20 to 30 minutes, and the whole session, including deploy, the run and pushing, roughly 30 to 45 minutes.

RunPod bills per second. On-demand A100 prices on runpod.io/pricing (page updated 2026-10-08):

| GPU | Secure Cloud | Community Cloud |
|---|---|---|
| A100 SXM 80 GB | $1.79/hr | $1.39/hr |
| A100 PCIe 80 GB | $1.79/hr | $1.19/hr |

At $1.79/hr, a 45-minute session is about $1.35; a failed first attempt plus a full rerun stays under $3. The default disks (80 GB at $0.10/GB/month while running) add about a cent per hour, and nothing once the pod is terminated.
