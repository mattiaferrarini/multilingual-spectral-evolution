# CS-552 Project — RCP Quick Start

Interactive Jupyter Lab on an A100 with PyTorch, vLLM, and the Hugging
Face stack already installed. No Docker building required for most teams.
The included `Dockerfile` and `build.sh` are only a base for teams that
genuinely need to make their own custom image.

For the official project scope, grading weights, rubrics, report
requirements, and deadlines, refer to the
[CS-552 Open Project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit).
This README focuses on the compute environment and the files needed to
run and grade the code.

Your milestone notebooks must live in your project repo, under
`individual_notebooks/`, and be committed with your submission. Use
`/scratch` for caches, datasets, checkpoints, and large generated files,
not as the only place where deliverable notebooks exist.

---

## TL;DR

1. Connect to the **EPFL VPN**, then install the Run:AI CLI and `runai login`.
2. Open `submit.sh`. **Set `GASPAR="gaspar"`** to your EPFL username
   and **set `GROUP="gXX"`** to your team number — both are mandatory;
   the script refuses to run otherwise.
3. `./submit.sh`
4. Connect to the pod (pick one):
   - **Jupyter:** wait until the job is `Running`, then
     `runai port-forward <job-name> --port 8888:8888`, and open
     `http://localhost:8888` (token: `cs552`)
   - **Shell:** `runai bash <job-name>`
   - **VS Code:** attach via the Kubernetes extension — see below.
5. **When you stop working: `runai delete job <job-name>`.**

---

## One-time setup

1. **Connect to the EPFL VPN.** You must be on the EPFL VPN to submit
   jobs to the cluster.
2. **Install the Run:AI CLI** and log in with your EPFL credentials. See
   the [RCP docs](https://docs.rcp.epfl.ch) if you haven't done this
   before.
3. **Set your project context** (replace `<gaspar>` with your username):
   ```bash
   runai config project course-cs-552-<gaspar>
   ```
   `submit.sh` also passes this project explicitly when submitting the
   job, so it does not depend on any other default Run:AI project.
4. **Edit `submit.sh`** — set `GASPAR="gaspar"` to your EPFL username
   (e.g. `jdupont`) and `GROUP="gXX"` to your team number (e.g. `g07`).
   **Required.** The script refuses to run with either placeholder.
5. *(Optional)* Export tokens in your shell so jobs pick them up:
   ```bash
   export HF_TOKEN=hf_xxx
   export WANDB_API_KEY=xxx
   ```

## Launch a job

Each job runs on **1 GPU (40GB A100)** — the course cap for this
setup. Asking for more leaves the job stuck `Pending`.

```bash
./submit.sh           # default
./submit.sh train     # custom job suffix
```

Wait for `Running`:
```bash
runai describe job <job-name>
```

In a second terminal, forward the port:
```bash
runai port-forward <job-name> --port 8888:8888
# Open http://localhost:8888 — token is "cs552"
```

This port-forward command is run after the job exists. Do not add
`--service-type portforward` to `runai submit`; some Run:AI CLI versions
reject it because `port-forward` is a client-side command.

When you're done **(read this please)**:
```bash
runai delete job <job-name>
```

## Connecting to your pod

You have three ways to interact with a running pod. Use whichever fits
the task.

### 1. Jupyter Lab (the default)

Already covered above — wait until the job is `Running`, run
`runai port-forward`, then open `http://localhost:8888`. Best for
notebook-driven exploration, plots, and the milestone deliverables.

### 2. A shell in the pod (`runai bash`)

For quick CLI work — running scripts, checking GPU usage, installing
packages, debugging — you don't need Jupyter at all:

```bash
runai bash <job-name>
```

You're now inside the container with a normal shell. Useful examples:

```bash
nvidia-smi                              # check GPU state
df -h /scratch                          # how much scratch space is left
python my_script.py                     # run something quickly
pip install some-extra-package          # ad-hoc install for this session
```

You can have a Jupyter port-forward running in one terminal *and* a
`runai bash` open in another, on the same pod.

### 3. VS Code attached to the pod

If you prefer VS Code over Jupyter for editing code, you can attach VS
Code directly to your running pod and edit files inside it as if they
were local. Full setup guide from RCP:
<https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-vscode>

Short version:

1. **Install VS Code** from <https://code.visualstudio.com>.
   The official Microsoft build is required — VSCodium does **not**
   work with the Kubernetes attachment flow.
2. **Install two extensions** from the VS Code Marketplace, both from
   Microsoft:
   - [Kubernetes](https://marketplace.visualstudio.com/items?itemName=ms-kubernetes-tools.vscode-kubernetes-tools)
   - [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
   Be careful — there are several Kubernetes extensions on the
   Marketplace; only the official Microsoft one is supported.
3. **Set up your kubeconfig** at `~/.kube/config` per the
   [RCP Quick Start](https://wiki.rcp.epfl.ch/home/CaaS/Quick_Start).
   The Kubernetes extension reads it automatically.
4. **Attach to your pod.** Click the Kubernetes icon in the left
   sidebar, expand your namespace under the cluster, find your running
   pod (named `<job-name>-0-0`), right-click → **Attach Visual
   Studio Code**. A new VS Code window opens, connected to the pod.
   The bottom-left status bar shows which pod you're attached to.
5. **Open files**: File → Open Folder, then type a path inside the pod
   (e.g. `/scratch`). You can edit files in place.
6. **Open a terminal**: Terminal → New Terminal opens a shell in the
   pod, same as `runai bash`.

VS Code is the most ergonomic option for serious code editing during
the project. Jupyter is still the right tool for the deliverable
notebooks.

## What's in the image

The course image (`registry.rcp.epfl.ch/course-cs-552/base-vllm:v1`) is
built on the official vLLM image and ships:

- **Core**: CUDA, PyTorch, vLLM, FlashAttention, FlashInfer, bitsandbytes
- **Training**: transformers, TRL (SFT/DPO/PPO), PEFT (LoRA/QLoRA), accelerate
- **Data**: datasets, huggingface_hub with hf_transfer
- **RAG**: sentence-transformers, faiss-cpu, rank-bm25, langchain
- **Eval**: lm-eval-harness, rouge-score, sacrebleu, bert-score
- **Tracking**: wandb, tensorboard
- **Notebook**: jupyterlab with widgets

## I need a package that isn't in the image

Three options, in order of preference:

1. **`pip install` from a notebook cell** — works for the session, takes
   seconds. Fine for one-off experiments.
2. **`requirements.txt` in your repo** — keep a `pip install -r
   requirements.txt` cell at the top of your notebook. Works for the
   grader too.
3. **Build your own image.** If your project genuinely needs something
   that can't be pip-installed (custom CUDA kernels, weird system libs), you can use the included `Dockerfile` and `build.sh` as a base, then build and push your own image to the same Harbor project (`registry.rcp.epfl.ch/course-cs-552/{your-image-name}:{tag}`). The initial build can take a long time (about 30 minutes), so plan for that.

Stick with the course image unless you have a concrete reason to build a
custom one.

## Storage layout inside the pod

| Path | What it is | Access |
|---|---|---|
| `/scratch` | Team scratch | shared with your group, RW |
| `/shared-ro/datasets` | Course datasets | read-only, all students |
| `/shared-ro/models` | Course base models | read-only, all students |
| `/shared-rw` | Course-wide writable scratch | RW for **everyone** — careful |

**Use `/scratch` for everything heavy** — clone your repo there, save
model checkpoints, store the HF cache, log wandb runs. The HF cache is
already pointed at `/scratch/hf_cache`, so `from_pretrained()` will
download there automatically and your teammates will see the cached
files.

Keep deliverable notebooks in your repo, not as loose files in
`/scratch`. It is fine to edit a repo clone that lives under `/scratch`,
but the notebook files must be committed in `individual_notebooks/` so
the graders get them from a clean clone.

> ⚠️ `/shared-rw` is writable by **all 285 students**. Don't put anything
> sensitive there, and don't rely on files in it persisting — anyone can
> overwrite or delete them.

> ℹ️ Anything in `/scratch` will be wiped end of July 2026.

## GPU etiquette (please read)

The course has **75 A100s shared across ~285 students**. The scheduler
caps each allocation at 1 GPU at a time, but doesn't otherwise prevent
you from holding it indefinitely. A few habits keep things working for
everyone, especially around the May 24 and June 7 deadlines:

- **Delete idle Jupyter jobs.** If you walk away from your laptop for
  more than ~30 minutes, `runai delete job <name>`. You can resubmit in
  ~5 seconds when you come back.
- **Use `--interactive` for exploration, not for long training.** This
  is what `submit.sh` does by default. Interactive jobs are preemptible,
  so the scheduler can reclaim them when capacity is tight — that's the
  right behavior for a notebook session.
- **For long final training runs, submit a non-interactive training
  job** (a separate script — ask your mentor for the pattern). Those are
  non-preemptible and won't be killed mid-epoch, but you also can't sit
  on them while idle.
- **Expect queues during deadline week.** Plan compute-heavy work
  earlier, not the night before.

## Common patterns

**Load a model** (caches to `/scratch/hf_cache`, downloaded once per group):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype="auto", device_map="auto"
)
```

**Run vLLM**:
```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.2-3B", download_dir="/scratch/hf_cache")
out = llm.generate(["Hello, "], SamplingParams(max_tokens=64))
```

**Fine-tune with TRL** — point output to `/scratch/runs/<exp>`:
```python
from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(output_dir="/scratch/runs/sft-v1", ...)
```

## What you turn in
 
The starter `submit.sh` in this repo is what TAs will run to grade your
work. The full deliverable checklist and rubrics are in the
[Open Project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit);
the items below are the code/notebook requirements relevant to this
repository.
 
**Milestone 2 — Preliminary Results (May 24)**
- `submit.sh` (this file, with `GASPAR` set to your EPFL username and
  `GROUP` set to your team number)
- `individual_notebooks/<first>_<last>.ipynb` — one notebook per
  teammate, demonstrating your individual contribution so far. The
  notebook must be in the repo, committed, and run end-to-end without
  errors in the environment that `submit.sh` launches.

**Milestone 3 — Final Submission (June 7)**
- `submit.sh` (same file, kept up to date with your `GASPAR` and
  `GROUP`)
- `individual_notebooks/<first>_<last>.ipynb` (one per teammate, deeper
  analysis — error analysis, ablations, attention viz, etc.). These
  notebooks must be in the repo, committed, and not only saved somewhere
  in `/scratch`.
- The 4-page report and full project code, per the project handout.
In both cases, your notebooks should preferably load models and datasets from
Hugging Face, and **must run inside the
pod produced by your `submit.sh`** — that's how TAs grade them.
For proposal, literature review, progress report, final report, and
rubric details, use the Open Project description as the source of truth.
 
### Modifying `submit.sh`
 
You can either ship the starter `submit.sh` as-is (recommended), or
modify it if your project genuinely needs something different — for
example pointing at a custom image you built, mounting additional
PVCs, or changing environment variables. **Both are allowed.**
 
> ⚠️ If you modify `submit.sh` and it doesn't work, you get **zero
> code** (i.e., individual notebook points) for that milestone. 
> When grading the project, the TAs will not debug your `submit.sh` 
> or your image. You are responsible for making sure it works from a clean
> clone of your repo, and that it launches a working pod where your
> notebooks execute without errors. 
 
So:
 
- **Default path** (most teams): leave `submit.sh` alone except for
  the `GASPAR="gaspar"` and `GROUP="gXX"` lines. The course image
  already covers fine-tuning, inference, RAG, and evaluation.
- **Custom path** (advanced teams): if you change the image, mounts,
  or anything else, test that a clean clone of your repo + your
  modified `submit.sh` actually launches a working pod and your
  notebooks execute. Run it yourself from a fresh terminal before
  submitting.
Use `/scratch` for paths, not personal absolute paths like
`/home/<your-gaspar>/...`, so the grader sees the same files in the
same places you do.
