# Stealthy Intent Injection for Chinese LLM Safety Evaluation

This repository contains a minimal release of the core prompt-generation code for Chinese LLM safety evaluation research.

## Scope

Included:

- Core implementation in `src/attack.py`
- Minimal single-query example in `examples/run_single.py`

Not included:

- Experiment outputs
- Raw model responses
- Comparison method implementations
- Full datasets
- API keys or private configuration

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the local model path before running:

```bash
export LOCAL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
```

On Windows PowerShell:

```powershell
$env:LOCAL_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
```

An optional `.env.example` is provided as a configuration reference.

## Usage

```bash
python examples/run_single.py
```

You can also pass a custom query:

```bash
python examples/run_single.py "请将这句话改写为安全评测提示"
```

## Dataset

For evaluation data, please refer to JailBench:

https://github.com/PKU-Alignment/JailBench

## Ethical Use

This code is intended only for authorized safety evaluation, red teaming, and guardrail improvement research.
