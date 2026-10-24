"""
End-to-end benchmark: a HuggingFace LLaMA-architecture model, stock versus
patched with this kit's kernels (RMSNorm, SwiGLU, RoPE). The model runs with
its default attention (SDPA), which fuses the softmax into the attention
kernel, so the online softmax is not part of this benchmark; it is covered
end to end by tests/test_e2e_llama.py under eager attention.

Reports, per variant:
  * prefill tokens/s: one forward over a prompt of --prompt-len tokens.
  * decode tokens/s: `model.generate` producing --new-tokens tokens with the
    KV cache; the measured prefill time is subtracted and the remaining
    time is divided by (new_tokens - 1) decode steps, i.e. the steady-state
    per-token rate including generate's own Python overhead.
Each number is the median of --iters runs after one warm-up run, timed with
CUDA events around the call.

What to expect, stated up front so the table is read honestly: at batch 1,
decode is bound by streaming the model weights once per token (2.2 GB in
fp16 for a 1.1B model) plus per-kernel launch overhead, and prefill is
dominated by the cuBLAS GEMMs. The three ops this kit replaces are a small
fraction of a layer's bytes and FLOPs, so the patched/stock ratio measures
what those ops cost in the stock implementation, not a model-wide claim.

Model: $MODEL (default TinyLlama/TinyLlama-1.1B-Chat-v1.0), loaded once per
--dtypes entry (default float16). Results go into the same
benchmarks/results/<gpu>_<date>.json as the kernel benchmarks, one row per
dtype under the "e2e" key.

    python benchmarks/bench_e2e.py --dtypes float16 bfloat16
    MODEL=meta-llama/Llama-3.2-1B python benchmarks/bench_e2e.py
"""

import argparse
import os
import statistics

import torch

from _common import DTYPE_MAP, require_cuda, resolve_peak_gbps, run_metadata, save_section
from triton_inference_kit.llama_patch import assert_llama_architecture, patched_llama

DEFAULT_MODEL = os.environ.get("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def timed_s(fn, iters: int) -> float:
    fn()  # warm-up (also pays any Triton / CUDA JIT compile)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1e3)
    return statistics.median(samples)


def measure(model, input_ids, attention_mask, new_tokens: int, iters: int, eos_id: int) -> dict:
    B, S = input_ids.shape
    with torch.no_grad():
        prefill_s = timed_s(lambda: model(input_ids, attention_mask=attention_mask), iters)
        generate_s = timed_s(
            lambda: model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
            ),
            iters,
        )
    decode_s = max(generate_s - prefill_s, 1e-9)
    return {
        "prefill_s": prefill_s,
        "generate_s": generate_s,
        "prefill_tok_s": B * S / prefill_s,
        "decode_tok_s": B * (new_tokens - 1) / decode_s,
    }


def print_table(variants: dict) -> None:
    stock = variants["stock"]
    header = f"{'variant':>22} | {'prefill tok/s':>14} | {'decode tok/s':>13} | {'prefill vs stock':>16} | {'decode vs stock':>15}"
    print(header)
    print("-" * len(header))
    for name, v in variants.items():
        print(
            f"{name:>22} | {v['prefill_tok_s']:>14.1f} | {v['decode_tok_s']:>13.1f} | "
            f"{v['prefill_tok_s'] / stock['prefill_tok_s']:>15.3f}x | "
            f"{v['decode_tok_s'] / stock['decode_tok_s']:>14.3f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or path (or set $MODEL).")
    parser.add_argument("--dtypes", nargs="+", default=["float16"], choices=["float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rmsnorm", choices=["triton", "cuda", "both"], default="both",
                        help="Which RMSNorm implementation(s) to bench in the patched model.")
    parser.add_argument("--peak-gbps", type=float, default=None, help="Only recorded in the metadata.")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    require_cuda()
    from transformers import AutoModelForCausalLM

    try:
        peak_gbps, peak_source = resolve_peak_gbps(args.peak_gbps)
    except SystemExit:
        peak_gbps, peak_source = None, "unknown"
    meta = run_metadata(peak_gbps, peak_source)
    print(f"GPU: {meta['gpu']} | driver {meta['driver']} | CUDA {meta['cuda']} | "
          f"torch {meta['torch']} | triton {meta['triton']}")

    impls = ["triton", "cuda"] if args.rmsnorm == "both" else [args.rmsnorm]
    rows = []
    for dtype_str in args.dtypes:
        print(f"Loading {args.model} in {dtype_str} ...")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE_MAP[dtype_str]).to("cuda").eval()
        assert_llama_architecture(model)
        eos_id = model.config.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0]

        # Random token ids are fine for a throughput measurement: the work per
        # token does not depend on which tokens they are.
        torch.manual_seed(0)
        input_ids = torch.randint(0, model.config.vocab_size, (args.batch, args.prompt_len), device="cuda")
        attention_mask = torch.ones_like(input_ids)

        variants = {}
        print("stock ...")
        variants["stock"] = measure(model, input_ids, attention_mask, args.new_tokens, args.iters, eos_id)
        for impl in impls:
            name = f"patched (rmsnorm={impl})"
            print(f"{name} ...")
            with patched_llama(rmsnorm_impl=impl) as stats:
                variants[name] = measure(model, input_ids, attention_mask, args.new_tokens, args.iters, eos_id)
            variants[name]["kernel_calls"] = dict(stats.calls)

        print(f"\n=== e2e: {args.model}, {dtype_str}, batch={args.batch}, prompt_len={args.prompt_len}, "
              f"new_tokens={args.new_tokens}, median of {args.iters} ===")
        print_table(variants)

        rows.append({
            "label": f"{args.model} {dtype_str}",
            "model": args.model,
            "dtype": dtype_str,
            "batch": args.batch,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "iters": args.iters,
            "transformers": __import__("transformers").__version__,
            "variants": variants,
        })
        del model
        torch.cuda.empty_cache()

    kwargs = {"results_dir": args.results_dir} if args.results_dir else {}
    save_section("e2e", rows, meta, **kwargs)


if __name__ == "__main__":
    main()
