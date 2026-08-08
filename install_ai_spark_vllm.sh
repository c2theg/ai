#!/usr/bin/env bash
# Christopher Gray  |  Version: 0.3.28  |  Update: 8/8/2026
# vLLM install, model download, and serve script for DGX Spark / NVIDIA systems
#
# Update Yourself:
#   curl -fsSL -o 'install_ai_spark_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark_vllm.sh' && chmod u+x install_ai_spark_vllm.sh
#   ./install_ai_spark_vllm.sh --start "Qwen3.6-35B-A3B-NVFP4:8011,Qwen3-Reranker-4B:8010"
#   ./install_ai_spark_vllm.sh --start "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16:8006,Qwen3-Reranker-4B:8010"
#   
#   ./install_ai_spark_vllm.sh --start "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8:8018,Qwen3-Reranker-4B:8010"
#   ./install_ai_spark_vllm.sh --start "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4:8019,Qwen3-Reranker-4B:8010"

#   ./install_ai_spark_vllm.sh --start "Qwen3.6-35B-A3B-FP8:8006,Qwen3-Reranker-4B:8010"
#   ./install_ai_spark_vllm.sh --start "Qwen3.6-35B-A3B-NVFP4:8006,Qwen3-Reranker-4B:8010"
#   ./install_ai_spark_vllm.sh --start "Qwen3.6-35B-A3B-NVFP4:8006"
#   ./install_ai_spark_vllm.sh --start "Qwen3.6-35B-A3B-NVFP4:8010,Qwen3.6-27B-NVFP4:8011"
#   ./install_ai_spark_vllm.sh --start "Gemma-4-31B-IT-NVFP4:8001"
#
# Move to DGX Spark / GB10:
#   scp install_ai_spark_vllm.sh root@<dgx-ip>:/home/user/install_ai_spark_vllm.sh
#   
#   scp install_ai_spark_vllm.sh user@10.11.1.10:/home/user/install_ai_spark_vllm.sh
#
# Usage:
#   ./install_ai_spark_vllm.sh              — full install: packages, docker, venv, download, serve
#   ./install_ai_spark_vllm.sh --serve-only — skip install/download; jump straight to model serve
#   ./install_ai_spark_vllm.sh -s           — same as --serve-only
#
#   Headless / boot-time serving (no prompts — safe for cron):
#   ./install_ai_spark_vllm.sh --start <spec>        — start 1+ models non-interactively
#       <spec> = model[:port][,model[:port]...]      — model = local dir name, HF repo id,
#                                                      or catalog index. --start is repeatable.
#       e.g.  --start Qwen3.6-35B-A3B-NVFP4
#             --start "Qwen3.6-35B-A3B-NVFP4,Qwen3-Reranker-4B"
#   Boot-time auto-start (persisted — no crontab edit needed to change models):
#   ./install_ai_spark_vllm.sh --set-boot-model <spec> — save <spec> as the model(s) that
#                                                        start at boot (validated, not yet
#                                                        started). Safe to run any time,
#                                                        repeatedly — never touches crontab.
#   ./install_ai_spark_vllm.sh --install-cron [spec]   — install the (one-time, stable)
#                                                        @reboot entry that runs --start-saved.
#                                                        Pass a spec to set it in the same step;
#                                                        omit it to install using whatever is
#                                                        already saved via --set-boot-model.
#   ./install_ai_spark_vllm.sh --start-saved           — headless-start whatever spec is
#                                                        currently saved (what @reboot runs;
#                                                        also handy to test the boot model now).
#   ./install_ai_spark_vllm.sh --remove-cron           — remove the @reboot entry (the saved
#                                                        spec itself is left alone)
#   ./install_ai_spark_vllm.sh --list-models           — print the servable-model catalog and exit
#   ./install_ai_spark_vllm.sh --health                — report which served models are up, and exit
#
#   Ports: served models get SEQUENTIAL ports in launch order starting at
#   BASE_PORT (default 8006) — 1st model → 8006, 2nd → 8007, and so on — so the
#   same launch order always yields the same ports. Pin a specific model with
#   "model:PORT" (e.g. --start "Qwen3-Reranker-4B:8021"); pinned ports are kept
#   and skipped over by the sequential counter. Before binding, the script checks
#   whether a port is already in use, shows what holds it, and offers to kill it
#   (interactive) or reclaims it only from a prior vLLM process (headless).
#
# ── Changelog ─────────────────────────────────────────────────────────────────
#
# v0.3.28  8/8/2026
#   - Reorganized the catalog's menu grouping: the "General"/"Coding"/"Reasoning"
#     categories (which mixed task labels with no consistent architecture
#     meaning) are replaced by "MoE Models" and "Dense Models", classified by
#     actual architecture — an "-A#B" active-param suffix (A3B/A4B/A10B/A12B)
#     means MoE, its absence means dense. Reclassified: Qwen3.6-35B-A3B-FP8,
#     NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4, Qwen3-Coder-30B-A3B-Instruct,
#     gemma-4-26B-A4B-it, Nemotron-3-Nano-Omni-30B-A3B-Reasoning (BF16/FP8/
#     NVFP4), Qwen3.6-35B-A3B-NVFP4(-Fast) → "MoE Models"; DeepSeek-R1-Distill-
#     Qwen-32B, gemma-4-31B-it, Qwen3.5-4B/2B/9B, Qwen3.6-27B-NVFP4, Gemma-4-31B-
#     IT-NVFP4 → "Dense Models". Left "Super Large" (needs its own memory-
#     warning logic — _check_vram, _serve_model, _vllm_launch all match on that
#     exact string), "Embeddings", "Reranking", and "ASR" untouched — those are
#     either non-chat task types or functionally load-bearing, not just display
#     labels. No catalog indices moved (still cosmetic-only — see the comment
#     above _checkbox_menu's sort block); only MDL_CAT values and the menu's
#     sort-rank case statement changed.
#
# v0.3.27  8/8/2026
#   - Decoupled the @reboot boot model from the crontab entry itself. Previously
#     --install-cron <spec> baked the spec directly into the crontab line, so
#     changing which model starts at boot meant re-running --install-cron with
#     a new spec (editing crontab) every time. Now: --set-boot-model <spec>
#     validates and writes the spec to BOOT_MODEL_SPEC_FILE
#     ($BASE_DIR/.boot-model-spec) — a plain file write, no crontab touched, safe
#     to call repeatedly. --install-cron installs a STABLE @reboot line that
#     always runs the new --start-saved (reads BOOT_MODEL_SPEC_FILE and heads
#     into the normal --start flow) — that line never needs to change again.
#     --install-cron [spec] still accepts an optional spec as a shorthand for
#     --set-boot-model + --install-cron in one step; called with no args it
#     (re)installs using whatever is already saved, erroring if nothing is.
#     --remove-cron is unchanged (removes the crontab entry only; the saved
#     spec file is left alone so re-running --install-cron later needs no spec).
#     Extracted the shared spec-validation logic (was duplicated inline in
#     _install_boot_cron) into _validate_spec_or_die, used by both
#     _install_boot_cron and the new _set_boot_model.
#
# v0.3.26  8/8/2026
#   - Replaced nvidia/Qwen3.6-35B-A3B-NVFP4's serve profile with a fuller DGX
#     Spark config: dropped the explicit --quantization modelopt_fp4 (now
#     auto-detected, needed for --moe-backend marlin to apply), added
#     --attention-backend flashinfer, --moe-backend marlin, --async-scheduling,
#     --load-format fastsafetensors, MTP --speculative-config (3 draft tokens,
#     triton MoE backend), and switched the tool/reasoning parsers from hermes
#     to qwen3_xml/qwen3. gpu-memory-utilization raised 0.30 → 0.4 and
#     max-model-len raised 32768 → 262144 (full context) — this profile is no
#     longer sized to co-run with the 27B NVFP4 model at the same time; catalog
#     VRAM estimate bumped 22 → 49 GB to match. Also added
#     sjug/Qwen3.5-122B-A10B-NVFP4-resharded (catalog idx new, port 8034,
#     ~71 GB disk / ~72 GB VRAM, "Qwen3.5-122B-A10B-NVFP4-spark" local dir,
#     "Super Large") — same weights as RedHatAI's Qwen3.5-122B-A10B NVFP4
#     quantization, resharded by the uploader into 16 x ~4.7GB shards
#     specifically to avoid memory-allocation failures on DGX Spark's 128GB
#     unified pool. Serve profile pulled from the model card's DGX-Spark-
#     optimized command: --load-format fastsafetensors, --kv-cache-dtype fp8,
#     0.7 gpu-memory-utilization, full 262144 max-model-len, qwen3_coder
#     tool-call parser + qwen3 reasoning parser, and three VLLM_* env vars
#     (NVFP4_GEMM_BACKEND=marlin, TEST_FORCE_FP8_MARLIN=1,
#     MARLIN_USE_ATOMIC_ADD=1) that pin the Marlin GEMM backend the card
#     benchmarked ~2% faster than CUTLASS on GB10 — exported only around this
#     model's _vllm_launch call (prefix-assignment on the call, not a global
#     export), so it can't leak into any other model's launch.
#
# v0.3.25  8/1/2026
#   - The venv rebuild worked — flashinfer/cutlass-dsl/quack/torch/GPU preflight
#     all passed cleanly. Nemotron then hit a completely different, non-
#     environment bug: "AssertionError: In Mamba cache align mode, block_size
#     (2128) must be <= max_num_batched_tokens (2048)". Nemotron-3-Nano is a
#     hybrid Mamba/attention architecture; with --enable-prefix-caching on, vLLM
#     computes a Mamba-aligned KV block_size from the model's state layout and
#     REQUIRES max_num_batched_tokens >= that value. None of the three
#     Nemotron-3-Nano-Omni catalog entries (BF16/FP8/NVFP4) or the plain
#     Nemotron-3-Nano-30B-A3B-NVFP4 entry set --max-num-batched-tokens, so
#     vLLM's own default (2048) was too small. Fixed all four with
#     --max-num-batched-tokens 4096 (clears the observed 2128 with headroom).
#   - Added a general auto-repair for this failure class in
#     _diagnose_and_repair: parses "block_size (N) must be <= max_num_batched_
#     tokens (M)" from the log, computes a safe value (N rounded up to the next
#     1024), and retries the SAME model with --max-num-batched-tokens overridden
#     — not a venv reinstall, since this is a launch-args problem. Covers future
#     Mamba-hybrid catalog entries or context-length changes that shift the
#     required block_size past whatever constant is hardcoded.
#   - This required a new mechanism: _VLLM_ARG_OVERRIDE lets a repair inject
#     extra CLI args into the retry (argparse keeps the last value of a repeated
#     flag, so an appended override wins over the catalog's own args). Verified
#     the array plumbing is nounset-safe for an EMPTY override too — bash's
#     `${arr[@]}` on a legitimately-empty array throws "unbound variable" under
#     `set -u` on bash <4.4; guarded with the `${arr[@]+"${arr[@]}"}` idiom at
#     every expansion site (this script's `set -uo pipefail` needs it even
#     though Ubuntu 24.04 ships bash 5.2, which alone would not hit this).
#
# v0.3.24  8/1/2026
#   - Found the real ThrMma culprit, from the engine traceback v0.3.22 started
#     printing. cutlass-dsl was never the mismatch: a clean reinstall at BOTH
#     vLLM's pin (4.6.0) and the latest PyPI release (4.6.1) still lacked
#     cutlass.cute.core.ThrMma, proving no published cutlass-dsl has that symbol.
#     The traceback named the actual cause: NVIDIA's `quack` kernel library
#     (imported by vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py)
#     references cute.core.ThrMma in a type annotation, evaluated at import
#     time — quack itself is the stale/mismatched package.
#   - Added _repair_symbol_consumer as the next playbook tier after
#     _repair_cutlass_stack: when the symbol-defining package checks out clean
#     at every version, parse the engine traceback for the last non-
#     infrastructure site-packages/<pkg>/ frame (the code that actually
#     REFERENCES the symbol), map it to its distribution name via
#     importlib.metadata.packages_distributions, and reinstall that fresh —
#     letting pip's resolver reconcile it against whatever cutlass-dsl is
#     already installed, rather than guessing versions ourselves. Verified
#     against the exact missing attribute, not a bare import.
#     Ladder is now: cutlass-dsl repair → symbol-consumer repair → retry →
#     full venv rebuild → final retry.
#
# v0.3.23  8/1/2026
#   - Final escalation tier: automatic venv rebuild. When a model dies with an
#     environment-shaped error (missing symbol / import error / invisible GPU —
#     explicitly NOT out-of-memory) and the targeted repair + retry did not fix
#     it, _rebuild_vllm_venv moves vllm-install aside to .broken-<timestamp>,
#     re-runs the DGX Spark vendor installer in the correct parent directory,
#     verifies the fresh venv (vllm imports AND torch sees the GPU), and retries
#     the model one final time. On any rebuild failure the old env is restored.
#     Gated by AUTO_REBUILD_VENV (default true) and a 24h loop-guard stamp so a
#     non-venv failure can't loop 15-minute builds on cron restarts.
#     Rationale: per-package surgery on a venv this far off the vendor state is
#     guesswork; the vendor build is the tested GB10 combination.
#   - FORCE_VLLM_REINSTALL now works with --start/--serve-only. It previously
#     lived only in the full-install branch, which those modes skip entirely —
#     the flag was unreachable from the normal restart command.
#
# v0.3.22  8/1/2026
#   - ThrMma, round three — with evidence this time. v0.3.21's clean rebuild ran,
#     verified, retried… and still crashed, which proves a PRISTINE
#     nvidia-cutlass-dsl[cu13]==4.6.0 does not provide cutlass.cute.core.ThrMma.
#     The orphan-files theory is dead: vLLM's ==4.6.0 pin is metadata-stale — the
#     code path that executes (flashinfer 0.6.14's cute-dsl kernels, own
#     requirement just >=4.5.0) was written against a newer cutlass API.
#   - _repair_cutlass_stack now parses the EXACT missing symbol from the crash
#     ("module 'X' has no attribute 'Y'") and verifies repairs with getattr —
#     `import cutlass.cute.core` passes in every version and proved nothing,
#     which is how v0.3.21 declared success on a broken install. If a clean
#     install at vLLM's pin still lacks the symbol, it escalates to the latest
#     cutlass-dsl and keeps it only when the symbol appears. Runtime
#     compatibility beats metadata: the override is recorded in
#     $VENV/.cutlass-dsl-override and _align_vllm_pins skips cutlass while it
#     exists, so the next preflight can't tug-of-war the fix back down.
#   - Repair also purges JIT/compile caches (~/.cache/flashinfer, ~/.cache/vllm,
#     /tmp/torchinductor_*) — generated code survives every pip operation and
#     keeps referencing whatever cutlass API existed when it was generated.
#   - _show_log_tail now prints the EngineCore traceback frames (ERROR lines
#     tagged [core.py:NNN]) — they name WHICH file raised, e.g. who is calling a
#     missing symbol. All day we have known WHAT was missing but never WHO
#     wanted it; that gap is why root-causing took multiple rounds.
#
# v0.3.21  8/1/2026
#   - Added a post-mortem playbook with ONE automatic retry. When a model dies
#     during load, _diagnose_and_repair matches the log against failure signatures
#     the script knows (corrupted cutlass-dsl install, flashinfer version skew,
#     GPU not visible), runs the matching repair, and relaunches the model once.
#     This replaces the pattern of a human reading the traceback and running the
#     fix by hand — the crash names the broken subsystem; fix it, retry.
#   - Added _repair_cutlass_stack for "module 'cutlass.cute.core' has no attribute
#     'ThrMma'" persisting across ALL version combinations. Root cause: repeated
#     in-place install/upgrade/downgrade of nvidia-cutlass-dsl leaves orphaned files
#     in the `cutlass` package dir (pip removes only what the outgoing RECORD lists),
#     so version-correct metadata sits on a mixed package directory and Python
#     imports the stale files. Repair: uninstall all five cutlass-dsl dists, DELETE
#     leftover cutlass* dirs from site-packages, reinstall vLLM's exact pin with
#     extras intact, verify `import cutlass.cute.core`.
#   - _align_vllm_pins: compare PUBLIC versions so torchaudio 2.11.0+cu130 satisfies
#     ==2.11.0 instead of being flagged as drift (the constraints file blocked any
#     damage, but the report was wrong); torch/torchvision/torchaudio now skipped
#     outright — the constraints file owns them.
#   - _vllm_launch now rotates the previous vllm-<port>.log to .log.old at launch.
#     Headless mode skips the interactive clean-start wipe, so logs accumulated
#     across runs and the root-cause extractor kept greping exceptions from OLD
#     crashes (the stale pids in earlier output). On a playbook retry the failed
#     log is likewise moved aside so the retry's diagnosis is clean.
#
# v0.3.20  8/1/2026
#   - Added _align_vllm_pins(): reconciles EVERY unsatisfied "==" pin vLLM declares,
#     not just flashinfer. Special-casing flashinfer was too narrow — the same drift
#     hit nvidia-cutlass-dsl (4.6.1 installed, with its cu13 libs stranded at 4.6.0,
#     while vllm 0.26.0 pins nvidia-cutlass-dsl[cu13]==4.6.0) and apache-tvm-ffi.
#     That drift surfaces as a missing symbol deep in engine startup —
#     "module 'cutlass.cute.core' has no attribute 'ThrMma'" — not as a version
#     error, which is what made it so hard to place.
#     Extras are preserved in the install spec (nvidia-cutlass-dsl[cu13]==4.6.0, not
#     the bare name); installing without the extra is exactly what leaves a split
#     base/cu13 install behind. Runs under the torch constraints file, before the
#     flashinfer pass so flashinfer still gets the final say on its own packages.
#     LIMITATION: only exact "==" pins are reconciled. Range requirements (vLLM's
#     setuptools<81, numba's numpy<2.5) are reported by `pip check` but not auto-
#     fixed, since resolving ranges risks more collateral than it prevents.
#
# v0.3.19  8/1/2026
#   - Corrected the flashinfer guard's core premise. flashinfer-cubin is an OPTIONAL
#     prebuilt-kernel cache, not a required sibling — so "align the trio" was the
#     wrong goal whenever vLLM's pin has no cubin wheel. vllm 0.26.0 pins
#     flashinfer-python==0.6.14 and nvidia-cutlass-dsl[cu13]==4.6.0; no cubin 0.6.14
#     was ever published because that install has no cubin. Downgrading
#     flashinfer-python to 0.6.13 to match a stale cubin (v0.3.14–0.3.18 behavior)
#     produced the next failure instead: 0.6.13 calls cutlass.cute.core.ThrMma, which
#     nvidia-cutlass-dsl 4.6.0 does not expose, killing EngineCore at model load.
#     The guard now REMOVES cubin and honors vLLM's pin, only laddering down if that
#     also fails. First launch then JIT-compiles kernels (slower once, then cached).
#
# v0.3.18  7/31/2026
#   - flashinfer reconciliation now uses a pip CONSTRAINTS file instead of --no-deps.
#     --no-deps was too blunt: it did stop pip replacing NVIDIA's DGX Spark torch, but
#     it also stopped pip adjusting flashinfer's SIBLING packages. Pinning
#     flashinfer-python down from 0.6.14 to 0.6.13 that way left nvidia-cutlass-dsl at
#     the version 0.6.14 wanted, and the engine then died at model-load time with
#       AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
#     — flashinfer calling a CuTe DSL API its sibling no longer exposed. Introduced by
#     the v0.3.14 guard; this is the fix. New _torch_constraints_file pins the
#     installed torch/torchvision/torchaudio (local segment stripped, so 2.11.0
#     still matches the 2.11.0+cu130 NVIDIA wheel) and lets the resolver correct
#     everything else.
#   - Torch rollback/restore still uses --no-deps deliberately: there the goal IS to
#     move torch alone without cascading into the rest of the stack.
#
# v0.3.17  7/31/2026
#   - Full install no longer re-runs the DGX Spark vendor installer when a working
#     vLLM venv already exists. That installer builds its venv at ./vllm-install
#     RELATIVE TO CWD and compiles Triton from source (~10 min, ~1 GB) — so running
#     the script from a new directory produced a second venv at a path the venv
#     search did not even look in, and the entire build was discarded. Override with
#     FORCE_VLLM_REINSTALL=true.
#   - Added "$PWD/vllm-install/.vllm" to the venv search list, so a venv the vendor
#     installer just created in the current directory is found instead of orphaned.
#   - Banner version is now parsed from the header comment instead of being a second
#     hand-maintained copy. It had drifted to "0.3.12 / 7/15/2026" while the header
#     was several versions ahead, making a freshly-copied script look stale.
#   - flashinfer: when vLLM pins a companion version that was never PUBLISHED, say so
#     instead of printing a remediation command that cannot succeed. vllm 0.26.0 pins
#     flashinfer-python==0.6.14 but flashinfer-cubin stops at 0.6.13 upstream, so the
#     trio correctly settles at 0.6.13 and the old advice told the user to run an
#     install that always fails. Detected via pip's "No matching distribution found".
#
# v0.3.16  7/31/2026
#   - flashinfer reconciliation now targets the version vLLM actually PINS, read
#     from vllm's own metadata (importlib.metadata.requires), instead of whatever
#     flashinfer-python happens to be installed. Aligning the trio to each other is
#     enough to make `import flashinfer` succeed, but it can settle on a version
#     vLLM does not want — observed in the wild as a venv self-consistent at 0.6.13
#     while vllm 0.26.0 required 0.6.14: it imported fine, served fine, and `pip
#     check` flagged it as broken. Falls back to the old behavior when vLLM declares
#     no pin. If the pinned version has no wheels, it ladders down (vLLM's pin →
#     installed flashinfer-python → a companion's version) and WARNS instead of
#     failing, since a self-consistent trio does run.
#   - _repair_gpu_driver no longer swallows modprobe's error. "Module nvidia not
#     found in directory /lib/modules/<kver>" / "Invalid module format" / "Key was
#     rejected by service" each name the cause outright; discarding that message is
#     what made this failure feel unexplainable.
#   - _repair_gpu_driver now distinguishes "no nvidia module exists for the running
#     kernel" (modinfo finds nothing) from a version mismatch, prints the running
#     kernel, and lists which kernels DO have a DKMS-built module — so "you booted a
#     kernel the driver was never built for" is visible rather than inferred.
#
# v0.3.15  7/31/2026
#   - GPU visibility guard for "RuntimeError: Failed to infer device type" (with
#     "Can't initialize NVML" / "0 active driver(s) found" / "No CUDA runtime is
#     found" above it). vLLM instantiates a default VllmConfig just to BUILD its CLI
#     parser, and DeviceConfig.__post_init__ probes for a device — so with no visible
#     GPU it dies during argument parsing, before any model or memory is involved.
#     New _preflight_gpu() splits the two root causes that log identically:
#       • DRIVER DOWN — nvidia-smi also fails. Attempts modprobe of the nvidia
#         modules, then compares the running driver (/proc/driver/nvidia/version)
#         against the installed one (modinfo); a mismatch means the module was built
#         for a different kernel and is reported with the dkms/reboot fix rather
#         than guessed at.
#       • TORCH BLIND — nvidia-smi works but the venv's torch has no CUDA runtime,
#         i.e. its wheel was replaced. Restored automatically from TORCH_STAMP.
#     nvidia-smi's exit status is what separates them. torch.version.cuda (build
#     metadata, driver-independent) is what distinguishes a clobbered wheel from a
#     driver outage — torch.cuda.is_available() cannot, since it is False for both.
#   - _maybe_update_vllm now snapshots torch before pip runs and rolls it back
#     automatically if the upgrade strips CUDA support (pip resolves torch against
#     PyPI, which does not carry NVIDIA's DGX Spark aarch64+CUDA build).
#   - AUTO_UPDATE_VLLM default flipped true → false. It ran an unpinned
#     `pip install -U vllm` on EVERY restart including cron @reboot, against the
#     exact build the script's own comments say not to replace. The rollback above
#     makes turning it back on much safer, but off is the right default here.
#   - Added AUTO_REPAIR_GPU (default true), AUTO_REPAIR_TORCH (default false — it
#     re-runs the vendor curl|bash installer, not something to fire unattended),
#     and TORCH_STAMP recording the last known-good CUDA-capable torch.
#
# v0.3.14  7/31/2026
#   - Fixed the flashinfer version-skew crash. flashinfer is three separately
#     versioned PyPI packages (flashinfer-python / -cubin / -jit-cache) that refuse
#     to import unless all three match. `pip install -U vllm` bumps flashinfer-python
#     only, so AUTO_UPDATE_VLLM silently broke the NEXT serve with "flashinfer-cubin
#     version (0.6.13) does not match flashinfer version (0.6.14)". vLLM imports
#     flashinfer from Sampler.__init__, so EngineCore died in init_device() — it
#     looked like an OOM (and sent us chasing --gpu-memory-utilization) but no GPU
#     memory had been allocated yet.
#     New _fix_flashinfer_versions() reconciles the trio automatically (aligns the
#     companions up to flashinfer-python, or pins flashinfer-python back down when
#     no matching companion wheel exists), using --no-deps so pip cannot clobber the
#     hardware-specific DGX Spark build. Called after every vLLM install/upgrade.
#   - Added _preflight_vllm_stack(), run before the serve section in all modes:
#     advisory `pip check` for dependency conflicts, the flashinfer self-heal, then
#     an `import vllm` gate. Environment breakage is reported once, up front, with
#     the real exception — instead of N identical 200-line EngineCore tracebacks
#     buried in per-port logs that the clean-start step then wipes.
#     The flashinfer result is gated separately from the vLLM import: `import vllm`
#     succeeds even with a broken flashinfer (vLLM only reaches for it later, in the
#     engine subprocess), so trusting the vLLM import alone would still green-light
#     a doomed serve.
#   - Added PREFLIGHT_STRICT (default true): a failed pre-flight aborts BEFORE the
#     clean-start block. That block kills every running vLLM process, so continuing
#     into a serve that cannot work would tear down healthy models and leave nothing
#     in their place. Override for one run with
#     PREFLIGHT_STRICT=false ./install_ai_spark_vllm.sh -s
#
# v0.3.12  7/15/2026
#   - Lowered the Gemma-4-31B-IT-NVFP4 serve profile for text-only use:
#     --language-model-only, 32768 context, 0.46 gpu-memory-utilization,
#     max-num-seqs 1, fp8 KV cache with scale calculation, prefix caching, and
#     chunked prefill. Pin it with :8001 when you want NVIDIA's example port.
#
# v0.3.11  7/15/2026
#   - Sequential served-model ports now start at 8006 instead of 8010. Headless
#     --start order already controlled launch order; the interactive serve menu
#     now also preserves the order models are toggled on, so the first chosen
#     model gets 8006, the second gets 8007, etc. Pinned model:PORT specs still
#     keep their explicit port and are skipped by the sequential allocator.
#
# v0.3.10  7/15/2026
#   - Added nvidia/Gemma-4-31B-IT-NVFP4 to the download/serve catalog. It uses
#     the NVIDIA ModelOpt NVFP4 vLLM profile. The later v0.3.12 profile lowers
#     context/utilization for text-only DGX Spark use.
#
# v0.3.7  7/14/2026
#   - Added nvidia/Qwen3.6-27B-NVFP4 to the download/serve catalog. It serves
#     with the current HF ModelOpt profile (modelopt v0.45 / NVFP4 1.0),
#     Qwen3 reasoning parser support, and a conservative 0.25
#     gpu-memory-utilization profile so it can co-run with
#     Qwen3.6-35B-A3B-NVFP4 (0.30) on a DGX Spark shared-memory pool. The HF
#     model card advertises up to 262144 context; this script defaults lower for
#     co-residency but exposes QWEN36_27B_MAX_MODEL_LEN to raise it for solo runs.
#
# v0.3.6  7/11/2026
#   - Added two quantized builds of the Omni reasoning model:
#     nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8  (port 8018, ~34 GB) and
#     nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 (port 8019, ~20 GB).
#     FP8 serves with --quantization modelopt @ 0.35 util; NVFP4 with
#     --quantization modelopt_fp4 @ 0.20 util. Both --trust-remote-code like the
#     BF16 build. Serve via --start with their local dir name or catalog index.
#
# v0.3.5  7/11/2026
#   - NVFP4 kernel-compile fix. On Blackwell (GB10, sm_121) FlashInfer JIT-compiles
#     ~29 CUTLASS GEMM kernels on first NVFP4/FP8 launch; ninja defaults to one
#     compile per core (~20) and each nvcc needs several GB, so the compiler was
#     OOM-killed ("Killed") and the engine died. Now: export MAX_JOBS (default 4)
#     + NVCC_THREADS=1 to cap parallelism; detect and export CUDA_HOME/PATH so
#     nvcc is found (and warn if it isn't); raise the readiness wait to
#     VLLM_READY_TIMEOUT (default 1800s) since the one-time compile can take
#     10-30 min (cached under /root/.cache/flashinfer afterward).
#
# v0.3.4  7/11/2026
#   - Auto-update vLLM (AUTO_UPDATE_VLLM=true): before serving, the script checks
#     PyPI for a newer vLLM and upgrades the venv if one is out (version-aware, so
#     it never downgrades a local nightly). Runs in every mode; best-effort so a
#     network error just logs and continues. Set false to pin the installed
#     version. Motivation: NVFP4 MoE checkpoints (Qwen3.6-35B-A3B-NVFP4) fail on
#     vLLM 0.20.2 with "KeyError: '…experts.w2_input_scale'" — newer vLLM adds the
#     modelopt NVFP4 expert-scale support.
#   - Failure diagnostics: _show_log_tail now extracts the real "ClassName:
#     message" exception line (last one, minus vLLM's generic wrapper) instead of
#     the earliest error-ish line, which had false-matched the huge INFO config
#     dump on substrings like device_config=cuda.
#
# v0.3.3  7/11/2026
#   - Qwen3-Reranker-4B is no longer pre-selected — DEFAULT_DL_INDICES and
#     DEFAULT_SERVE_INDICES are now empty, so nothing auto-starts; pick models in
#     the menus or with --start. Dropped the "★Default" label.
#   - RIGHT-SIZED --gpu-memory-utilization for the small models. The old values
#     were oversized on a false premise ("raise the fraction to clear the
#     all-process baseline"): the fraction is how much of the 121.7 GB pool THIS
#     model reserves and must be free at startup, so a big fraction on a small
#     model both wastes RAM and fails to start when memory is tight. The 4B
#     reranker at 0.55 was holding ~68 GB. New values (approx GB on a 121 GB box):
#       Qwen3-Reranker-4B 0.55→0.12 (~15), Qwen3-Embedding-4B 0.60→0.10 (~12),
#       bge-m3 0.45→0.06 (~7), bge-reranker-v2-m3 0.45→0.06 (~7),
#       Qwen3.5-4B 0.50→0.12 (~15), Qwen3.5-2B 0.45→0.08 (~10),
#       Qwen3.5-9B 0.75→0.20 (~24).
#   - Pre-flight memory check: before launching, the script compares the model's
#     reservation (fraction × total) against free RAM and, if it won't fit, prints
#     an actionable message (free memory or lower the fraction) instead of letting
#     vLLM crash with the cryptic "Free memory … less than desired" ValueError.
#     Headless skips a doomed launch; interactive asks before trying.
#
# v0.3.2  7/11/2026
#   - Predictable ports: served models are now assigned SEQUENTIAL ports in
#     launch order from BASE_PORT — so the same selection always maps to the
#     same ports (was: fixed per-model catalog
#     ports). Pin one with "model:PORT"; pinned ports are kept and skipped over.
#     A port map is printed before serving.
#   - Port-in-use guard: before a model binds its port, the script detects any
#     existing listener (lsof/ss/fuser), prints what it is, and — interactively —
#     asks whether to kill it; headless mode reclaims the port only if a prior
#     vLLM process holds it, else skips that model (never kills other services).
#   - Health check: after serving, an explicit UP / NOT-READY line per model with
#     its /v1/models test command and log path. New `--health` flag re-runs the
#     check standalone (probes catalog + BASE_PORT block); good for cron/monitoring.
#   - Auto-download (AUTO_DOWNLOAD=true): if a selected model is missing on disk
#     when it's time to serve it, the script downloads it first (HF CLI + HF_TOKEN)
#     instead of skipping — works in --serve-only and headless --start too.
#   - Failure diagnostics: on a model that dies during load, _show_log_tail now
#     prints a longer tail AND greps the whole log for the earliest error, since
#     vLLM's "Engine core initialization failed. See root cause above" hides the
#     real cause far above the shown traceback.
#   - Self-update line switched from `curl | bash` to `curl -o … && chmod` so it
#     replaces the saved local copy instead of executing the download.
#
# v0.3.0  7/11/2026
#   - Added nvidia/Qwen3.6-35B-A3B-NVFP4  (catalog idx 22, port 8016, ~22 GB)
#     and unsloth/Qwen3.6-35B-A3B-NVFP4-Fast (catalog idx 23, port 8017, ~22 GB).
#     NVFP4 (4-bit) halves the FP8 footprint — both use --quantization modelopt_fp4.
#   - Headless startup mode: --start "model[:port],..." serves 1+ models with NO
#     prompts (skips install, menus, memory prompts, docker containers, OpenWebUI).
#     Models are matched by local dir name, HF repo id, or catalog index; an
#     optional :port overrides the catalog port. Designed for cron @reboot.
#   - --install-cron <spec> writes the @reboot crontab entry for you (and
#     --remove-cron deletes it). --list-models prints the catalog and exits.
#   - REFACTOR: replaced the ~280 lines of per-index `if is_run_selected N`
#     serve blocks with one _serve_model() dispatching on the HF repo id, plus a
#     single loop over RUN_SELECTED. Serve args can no longer silently mis-map
#     when catalog indices shift (the v0.1.5 bug class is now impossible).
#   - BUGFIX (found during refactor): Qwen3.5-9B is really catalog idx 19 (its
#     _add line sits before the two appended ASR entries), but its serve block
#     checked idx 21 — so selecting 9B launched NOTHING, and selecting
#     Qwen3-ASR-1.7B (really idx 21) launched the 9B serve args. Fixed
#     automatically by the repo-id dispatch above.
#   - _vllm_launch: readiness poll now hits /health (cheaper than /v1/models)
#     every 5s instead of every 15s, printing progress every 30s — small models
#     become ready up to ~14s sooner; log noise unchanged.
#   - _kill_vllm_processes now also kills a previous run's sleep_watchdog.sh so
#     watchdogs no longer stack across restarts.
#   - Removed dead helpers is_dl_selected/is_run_selected (unused after refactor).
#
# v0.2.9  6/27/2026
#   - Per-model sleep prompt: replaced the single "Standard models timeout"
#     prompt with a prompt for every selected servable model. Default shown in
#     brackets is the catalog SLEEP_MIN override if set, else IDLE_SLEEP_MINUTES.
#     User can set a different idle-sleep timeout for each model before serving.
#
# v0.2.8  6/27/2026
#   - Sequential model startup: _vllm_launch now polls GET /v1/models after
#     launch and blocks until the model reports ready (or 12-minute timeout).
#     Each model fully loads before the next one starts, so later models see a
#     stable baseline when the KV-cache profiler runs — no more OOM from racing
#     concurrent startups. Also detects if the process dies mid-load and prints
#     the tail of the log immediately.
#   - Qwen3.5-9B gpu-memory-utilization raised 0.50 → 0.75 and max-model-len
#     capped at 16384. When Nemotron-3-Nano-Omni-30B (~62 GB) is already loaded,
#     0.75×121=90.75 GB budget leaves ~10 GB for KV cache after 9B weights.
#
# v0.2.7  6/26/2026
#   - Raised Qwen3-Reranker-4B gpu-memory-utilization 0.50 → 0.55. With more
#     models now loaded concurrently (4B/2B/9B small models), the KV-cache
#     profiler baseline crept to ~61.2 GB — just 0.71 GB over the 60.5 GB
#     budget at 0.50. New 0.55 budget (66.6 GB) gives ~5 GB KV headroom.
#
# v0.2.6  6/26/2026
#   - Fixed HF repo IDs for small Qwen3.5 dense models: Qwen/Qwen3.5-4B-Instruct
#     and Qwen/Qwen3.5-2B-Instruct → Qwen/Qwen3.5-4B and Qwen/Qwen3.5-2B
#     (the repos have no -Instruct suffix). Updated catalog, local dirs,
#     serve block --served-model-name args, and changelog refs.
#   - Fixed download loop to check exit code: now prints ❌ on failure instead of
#     falsely printing ✅ regardless of whether huggingface-cli succeeded.
#   - Added Qwen/Qwen3.5-9B (catalog idx 21, port 8015, ~18 GB VRAM)
#     with SLEEP_MIN=60 (same idle-sleep pattern as the 2B/4B small models).
#
# ──────────────────────────────────────────────────────────────────────────────

# ─── strict mode ──────────────────────────────────────────────────────────────
# -u: error on unset variables  -o pipefail: propagate pipeline failures
# -e (exit on error) is intentionally omitted — this script uses many
# [ cond ] && action patterns and || true guards that conflict with -e.
set -uo pipefail

# ─── argument parsing ─────────────────────────────────────────────────────────
SERVE_ONLY=0
HEADLESS=0        # set by --start / --start-saved: non-interactive serve, then exit
LIST_MODELS=0
HEALTH_CHECK=0    # set by --health: probe every servable catalog port, then exit
START_SPECS=""    # comma-separated model[:port] specs collected from --start
START_SAVED=0     # set by --start-saved: read the spec from BOOT_MODEL_SPEC_FILE
CRON_ACTION=""    # install | remove
CRON_SPEC=""
SET_BOOT_MODEL_SPEC=""   # set by --set-boot-model: persists the boot-time spec

while [ "$#" -gt 0 ]; do
    case "$1" in
        --serve-only|-s) SERVE_ONLY=1 ;;
        --start)
            [ "$#" -ge 2 ] || { echo "❌ --start needs a model spec, e.g. --start 'Model:8016,Model2'"; exit 1; }
            shift; START_SPECS="${START_SPECS:+$START_SPECS,}$1" ;;
        --start=*)       START_SPECS="${START_SPECS:+$START_SPECS,}${1#--start=}" ;;
        --start-saved)   START_SAVED=1 ;;
        --set-boot-model)
            [ "$#" -ge 2 ] || { echo "❌ --set-boot-model needs a model spec, e.g. --set-boot-model 'Model:8016'"; exit 1; }
            shift; SET_BOOT_MODEL_SPEC="$1" ;;
        --set-boot-model=*) SET_BOOT_MODEL_SPEC="${1#--set-boot-model=}" ;;
        --install-cron)
            # Spec is now OPTIONAL here: with one, it also updates the saved boot
            # model (like --set-boot-model); without one, it (re)installs the
            # @reboot entry using whatever spec is already saved.
            if [ "$#" -ge 2 ] && [ "${2:0:2}" != "--" ]; then
                shift; CRON_ACTION="install"; CRON_SPEC="$1"
            else
                CRON_ACTION="install"; CRON_SPEC=""
            fi ;;
        --install-cron=*) CRON_ACTION="install"; CRON_SPEC="${1#--install-cron=}" ;;
        --remove-cron)   CRON_ACTION="remove" ;;
        --list-models)   LIST_MODELS=1 ;;
        --health)        HEALTH_CHECK=1 ;;
        -h|--help)
            # Print the Usage block from the header (stop at the first Changelog line).
            sed -n '/^# Usage:/,$p' "$0" | sed -n '1,/^# ── Changelog/p' | sed '$d' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "⚠️  Unknown argument: $1 (see --help)" ;;
    esac
    shift
done

if [ -n "$START_SPECS" ]; then
    HEADLESS=1
    SERVE_ONLY=1   # headless mode never installs or downloads
fi

# Single source of truth for the version: parsed from the header comment on line 2.
# The banner used to carry its own hand-maintained copy, which drifted — it printed
# "0.3.12 / 7/15/2026" while the header and changelog were several versions ahead,
# so a freshly-copied script looked stale and there was no way to tell from the
# banner which build was actually running.
SCRIPT_VERSION=$(sed -n '2s/.*Version: *\([0-9][0-9.]*\).*/\1/p' "$0" 2>/dev/null)
SCRIPT_UPDATED=$(sed -n '2s|.*Update: *\([0-9][0-9/]*\).*|\1|p' "$0" 2>/dev/null)
[ -z "$SCRIPT_VERSION" ] && SCRIPT_VERSION="unknown"
[ -z "$SCRIPT_UPDATED" ] && SCRIPT_UPDATED="unknown"

echo "


 _____             _         _    _          _
|     |___ ___ ___| |_ ___ _| |  | |_ _ _   |_|
|   --|  _| -_| .'|  _| -_| . |  | . | | |   _
|_____|_| |___|__,|_| |___|___|  |___|_  |  |_|
                                     |___|

 _____ _       _     _           _              _____    __    _____
|     | |_ ___|_|___| |_ ___ ___| |_ ___ ___   |     |__|  |  |   __|___ ___ _ _
|   --|   |  _| |_ -|  _| . | . |   | -_|  _|  | | | |  |  |  |  |  |  _| .'| | |
|_____|_|_|_| |_|___|_| |___|  _|_|_|___|_|    |_|_|_|_____|  |_____|_| |__,|_  |
                            |_|                                             |___|


Version:  $SCRIPT_VERSION
Last Updated:  $SCRIPT_UPDATED

Update Yourself:
    curl -fsSL -o 'install_ai_spark_vllm.sh' 'https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_spark_vllm.sh' && chmod u+x install_ai_spark_vllm.sh

  YOU MUST HAVE A HUGGINGFACE ACCOUNT AND TOKEN TO DOWNLOAD MODELS!
    *** Update 'HF_TOKEN' on line 35 before running this script! ***
        Huggingface models:   https://huggingface.co/models


"

# =============================================
# CONFIGURATION — set these before running
# =============================================
HF_TOKEN=""                  # HuggingFace token — fallback if not in .env
                             # Get yours at: https://huggingface.co/settings/tokens
BASE_DIR="/opt/models"       # All paths derive from here — change this one line to relocate everything

MODELS_DIR="$BASE_DIR/vllm"           # Where all models will be downloaded
VLLM_VENV="$BASE_DIR/vllm-install/.vllm"  # venv created by the vLLM install script
NEMO_VENV="$BASE_DIR/nemo-venv"       # separate venv for NeMo ASR (avoids conflicts with vLLM)

# Persisted boot-time model spec, written by --set-boot-model and read by
# --start-saved. Decouples "what starts at boot" from the crontab entry itself:
# --install-cron writes a STABLE @reboot line (`--start-saved`) once, and
# changing which model boots after that is just --set-boot-model — no crontab
# edit, no re-running --install-cron. See _set_boot_model / _install_boot_cron.
BOOT_MODEL_SPEC_FILE="$BASE_DIR/.boot-model-spec"

# =============================================
# PREDICTABLE SERVE PORTS
# =============================================
# Served models are assigned sequential ports in launch order starting here:
# the 1st served model gets BASE_PORT, the 2nd BASE_PORT+1, and so on. This
# overrides each model's catalog port so the same launch order always yields the
# same ports. A per-model port pinned with `--start model:PORT` is respected and
# skipped over. Before binding, the script checks whether the port is already in
# use, shows what holds it, and (interactively) offers to kill it.
BASE_PORT=8006

# If a selected model isn't present on disk when it's time to serve it, download
# it automatically first (uses the HF CLI + HF_TOKEN). Applies to every mode,
# including --serve-only and headless --start. Set false to keep the old behavior
# of skipping a missing model instead.
AUTO_DOWNLOAD=true

# Before serving, check PyPI for a newer vLLM and upgrade the venv if one exists.
# Runs in every mode (including headless --start / cron). Best-effort: a network
# error or PyPI hiccup is logged and skipped, never blocking model startup.
# ⚠️  Trade-off: this reaches the network every run and a vLLM upgrade wheel can
# be large/slow, so a cron @reboot start may take longer. It also means a bad
# upstream release could regress a working stack — set false to pin the installed
# version once you have one that works (e.g. after NVFP4 support lands).
#
# ⚠️  DEFAULT FLIPPED TO false IN v0.3.15. On DGX Spark the venv's torch is an
# NVIDIA aarch64+CUDA build that PyPI does not carry. `pip install -U vllm` resolves
# its torch dependency against PyPI and can swap that build for a generic wheel with
# no CUDA runtime — after which torch sees no GPU and vLLM dies with "Failed to
# infer device type" before it can even parse arguments. Running that unpinned
# upgrade on EVERY restart (including cron @reboot) is a lot of exposure for a
# build you specifically do not want replaced. _maybe_update_vllm now also snapshots
# torch and rolls back automatically if an upgrade strips CUDA support, so turning
# this back on is far safer than it was — but it stays off by default.
AUTO_UPDATE_VLLM=false

# On startup, verify the GPU is actually visible and try to recover it if not.
# Two distinct failures, handled differently (see _preflight_gpu):
#   • driver down  — nvidia-smi fails / NVML won't init. Attempts a modprobe of the
#     nvidia modules; a kernel-vs-driver mismatch needs a DKMS rebuild or reboot and
#     is reported, not guessed at.
#   • torch blind  — nvidia-smi is fine but the venv's torch has no CUDA runtime,
#     i.e. the wheel got clobbered. Rolled back automatically when this script has a
#     recorded good torch version (see TORCH_STAMP); otherwise reported.
AUTO_REPAIR_GPU=true

# Where _maybe_update_vllm records the known-good torch version+CUDA before it
# touches pip, so a clobbered wheel can be restored without guessing.
TORCH_STAMP="$BASE_DIR/.torch-known-good"

# Final escalation: when a model dies with an environment-shaped error (missing
# symbol / import error / invisible GPU — NOT out-of-memory) and the targeted
# repair + retry did not fix it, rebuild the whole venv from the vendor installer
# and try once more. A fresh vendor build is the tested GB10 state; per-package
# surgery on a venv that has drifted this far is guesswork. The old env is moved
# aside (…/vllm-install.broken-<timestamp>), never deleted. A stamp file limits
# auto-rebuilds to one per 24h so a non-venv failure can't loop 15-minute builds.
AUTO_REBUILD_VENV="${AUTO_REBUILD_VENV:-true}"
REBUILD_STAMP="$BASE_DIR/.venv-rebuild-stamp"

# Last-resort repair for a clobbered torch when no TORCH_STAMP exists: re-run the
# vendor DGX Spark installer. OFF by default — it is a curl|bash from the network
# that rebuilds the venv and takes a while, which is not something to trigger
# unattended on a cron restart. Set true if you want fully hands-off recovery.
AUTO_REPAIR_TORCH=false

# Abort before the serve section when the pre-flight finds a venv that cannot
# possibly serve (skewed flashinfer trio, un-importable vLLM). Aborting early
# matters more than it sounds: the clean-start step kills every running vLLM
# process, so continuing into a doomed serve would tear down working models and
# replace them with nothing. Override for one run:
#   PREFLIGHT_STRICT=false ./install_ai_spark_vllm.sh -s
PREFLIGHT_STRICT="${PREFLIGHT_STRICT:-true}"

# Quantized models (NVFP4/FP8 on Blackwell sm_120/121) make FlashInfer JIT-compile
# ~29 CUTLASS GEMM kernels on first launch. ninja defaults to one compile per CPU
# core (~20), and each nvcc uses several GB — that OOM-kills the compiler
# ("Killed") and the engine dies. MAX_JOBS caps how many compile in parallel.
# Lower it (2, then 1) if you still see "Killed"; the compiled kernels are cached
# under /root/.cache/flashinfer, so this cost is paid only on the first launch.
MAX_JOBS=4

# How long _vllm_launch waits for a model to report ready before moving on. The
# first NVFP4/FP8 launch includes the one-time kernel compile above, which can
# take 10-30 min, so this is generous. Later launches (cache warm) are quick.
VLLM_READY_TIMEOUT=1800

# =============================================
# OPTIONAL FEATURES — toggle on/off
# =============================================
ENABLE_SEARXNG=true          # SearXNG web search engine for OpenWebUI (runs on port 4040)
SEARXNG_PORT=4040            # host port for SearXNG — change if 4040 is in use

BRAVE_SEARCH_API_KEY=""      # Brave Search API key — takes priority over SearXNG when set
                             # Get yours at: https://api.search.brave.com/

# =============================================
# OPENWEBUI AUTO-REGISTRATION
# =============================================
OWUI_ADMIN_EMAIL="admin@local"
OWUI_ADMIN_PASSWORD="Abc123!@#"

# =============================================
# MEMORY & AI INFRASTRUCTURE
# =============================================
IDLE_SLEEP_MINUTES=15        # Offload models to CPU after this many idle minutes (0 = disabled)
                             # Used as the fallback for any model without its own timeout.
STANDARD_SLEEP_MINUTES=15    # Default idle-sleep timeout (minutes) for all Standard models.
                             # The script prompts before serving so this can be overridden
                             # at runtime (0 = never sleep). Press Enter at the prompt to accept.

ENABLE_SQLITE_MEMORY=true    # SQLite DB for structured memory (exact facts, conversations)
SQLITE_RETENTION_DAYS=60     # Delete records older than N days  (≈ 2 months)
SQLITE_MAX_MB=100            # Also prune oldest rows when DB exceeds this size

ENABLE_QDRANT=true           # Qdrant vector DB for long-term semantic memory
QDRANT_HTTP_PORT=6333        # Qdrant REST API port
QDRANT_GRPC_PORT=6334        # Qdrant gRPC port
QDRANT_RETENTION_DAYS=60     # Delete vectors older than N days  (≈ 2 months)
QDRANT_MAX_GB=1              # Warn (and log) when storage exceeds this many GB

MEMORY_DIR="$BASE_DIR/memory"
SQLITE_DB="$MEMORY_DIR/structured_memory.db"
QDRANT_DATA_DIR="$BASE_DIR/qdrant"

# Load .env from same directory as this script — overrides tokens above if set there
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

_env_load() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d '=' -f2- | tr -d '"' | tr -d "'"; }
_env_save() {
    if grep -qE "^$1=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
    else
        echo "$1=$2" >> "$ENV_FILE"
    fi
}

ENV_HF_TOKEN=$(_env_load HF_TOKEN)
if [ -n "$ENV_HF_TOKEN" ]; then
    HF_TOKEN="$ENV_HF_TOKEN"; echo "✅ HF_TOKEN loaded from .env"
elif [ -n "$HF_TOKEN" ]; then
    _env_save HF_TOKEN "$HF_TOKEN"; echo "✅ HF_TOKEN saved to $ENV_FILE"
fi
[ -z "$HF_TOKEN" ] && echo "⚠️  HF_TOKEN is not set — gated models will fail."

ENV_BRAVE_KEY=$(_env_load BRAVE_SEARCH_API_KEY)
[ -n "$ENV_BRAVE_KEY" ] && BRAVE_SEARCH_API_KEY="$ENV_BRAVE_KEY" && echo "✅ BRAVE_SEARCH_API_KEY loaded from .env"
[ -z "$ENV_BRAVE_KEY" ] && [ -n "$BRAVE_SEARCH_API_KEY" ] && _env_save BRAVE_SEARCH_API_KEY "$BRAVE_SEARCH_API_KEY"

ENV_OWUI_EMAIL=$(_env_load OWUI_ADMIN_EMAIL)
[ -n "$ENV_OWUI_EMAIL" ] && OWUI_ADMIN_EMAIL="$ENV_OWUI_EMAIL" && echo "✅ OWUI_ADMIN_EMAIL loaded from .env"
[ -z "$ENV_OWUI_EMAIL" ] && [ -n "$OWUI_ADMIN_EMAIL" ] && _env_save OWUI_ADMIN_EMAIL "$OWUI_ADMIN_EMAIL"

ENV_OWUI_PASS=$(_env_load OWUI_ADMIN_PASSWORD)
[ -n "$ENV_OWUI_PASS" ] && OWUI_ADMIN_PASSWORD="$ENV_OWUI_PASS" && echo "✅ OWUI_ADMIN_PASSWORD loaded from .env"
[ -z "$ENV_OWUI_PASS" ] && [ -n "$OWUI_ADMIN_PASSWORD" ] && _env_save OWUI_ADMIN_PASSWORD "$OWUI_ADMIN_PASSWORD"

if [ -z "$OWUI_ADMIN_EMAIL" ] || [ -z "$OWUI_ADMIN_PASSWORD" ]; then
    echo "⚠️  OWUI credentials not set — visit http://localhost:3000 on first run to create your admin account."
fi

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CATALOG
# Fields: HF_REPO | LOCAL_DIR | DISPLAY_NAME | DISK_GB | VRAM_GB | PORT | CATEGORY | SLEEP_MIN(optional)
# VRAM_GB=0   → CPU/NeMo only — cannot be served via vLLM
# PORT=0      → download-only (ASR/NeMo)
# SLEEP_MIN   → optional per-model idle-sleep timeout in minutes; overrides the
#               global IDLE_SLEEP_MINUTES for this model only.  Omit to use global.
# ─────────────────────────────────────────────────────────────────────────────
MDL_HF=()
MDL_DIR=()
MDL_NAME=()
MDL_DISK=()
MDL_VRAM=()
MDL_PORT=()
MDL_CAT=()
MDL_SLEEP=()
MDL_PORT_EXPLICIT=()   # [idx]=1 when a --start "model:PORT" pinned this model's port
                       # (pinned models are exempt from sequential re-assignment)

_add() {
    local i=${#MDL_HF[@]}
    MDL_HF[$i]="$1"; MDL_DIR[$i]="$2"; MDL_NAME[$i]="$3"
    MDL_DISK[$i]="$4"; MDL_VRAM[$i]="$5"; MDL_PORT[$i]="$6"; MDL_CAT[$i]="$7"
    MDL_SLEEP[$i]="${8:-}"
}

# ── Standard models ────────────────────────────────────────────────────────────
# Sleep timeout: these all use STANDARD_SLEEP_MINUTES (prompted/overridable at
# runtime). The catalog leaves SLEEP_MIN blank here; it is filled in for the whole
# range below once the user confirms the timeout. The index span is captured in
# STANDARD_INDICES so the chosen value can be applied to exactly these models.
_STD_RANGE_START=${#MDL_HF[@]}
#        HF Repo                                                   Local Dir                               Display Name                          Disk VRAM  Port  Category
_add "Qwen/Qwen3.6-35B-A3B-FP8"                                  "Qwen3.6-35B-A3B-FP8"                   "Qwen3.6-35B-A3B (FP8)"                  35   38   8005  "MoE Models"
_add "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"               "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"  "Nemotron-3-Nano-30B-A3B (NVFP4)"        15   18   8006  "MoE Models"
_add "Qwen/Qwen3-Coder-30B-A3B-Instruct"                         "Qwen3-Coder-30B-A3B-Instruct"          "Qwen3-Coder-30B-A3B (BF16)"             60   65   8001  "MoE Models"
_add "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"                  "DeepSeek-R1-Distill-Qwen-32B"          "DeepSeek-R1-Distill-Qwen-32B (BF16)"    64   68   8002  "Dense Models"
_add "google/gemma-4-31B-it"                                     "gemma-4-31B-it"                        "Gemma 4 31B-it (BF16)"                  62   66   8009  "Dense Models"
_add "google/gemma-4-26B-A4B-it"                                 "gemma-4-26B-A4B-it"                    "Gemma 4 26B-A4B (BF16)"                 52   56   8007  "MoE Models"
_add "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" "Nemotron-3-Nano-Omni-30B (BF16)"  60   65   8008  "MoE Models"
_add "BAAI/bge-m3"                                               "bge-m3"                                "BGE-M3 (Embeddings)"                     3    4   8011  "Embeddings"
_add "Qwen/Qwen3-Embedding-4B"                                   "Qwen3-Embedding-4B"                    "Qwen3-Embedding-4B (Embeddings)"         8   10   8010  "Embeddings"
_add "BAAI/bge-reranker-v2-m3"                                   "bge-reranker-v2-m3"                    "BGE-Reranker-v2-m3 (Reranking)"          2    3   8020  "Reranking"
_add "Qwen/Qwen3-Reranker-4B"                                    "Qwen3-Reranker-4B"                     "Qwen3-Reranker-4B (Reranking)"            8    9   8021  "Reranking"
_add "nvidia/parakeet-tdt-0.6b-v3"                               "parakeet-tdt-0.6b-v3"                  "Parakeet-TDT-0.6B v3 (ASR / NeMo)"       1    0      0  "ASR"
_add "nvidia/nemotron-speech-streaming-en-0.6b"                  "nemotron-speech-streaming-en-0.6b"     "Nemotron-Speech-Streaming-0.6B (ASR)"    1    0      0  "ASR"

# Catalog indices of the Standard-models block above — used to apply the
# runtime-chosen sleep timeout to exactly these models.
STANDARD_INDICES=()
for _si in $(seq "$_STD_RANGE_START" $(( ${#MDL_HF[@]} - 1 ))); do STANDARD_INDICES+=("$_si"); done

# ── SUPER LARGE models (120B+ parameters) ─────────────────────────────────────
# Info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard
# Note: these require nearly the entire GPU — do not run alongside other large models.
# ⚠️  Verify HF repo IDs before downloading — these may require updated values.
_add "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"             "NVIDIA-Nemotron-3-Super-120B-A12B-BF16" "Nemotron-3-Super-120B-A12B (BF16) [SUPER]" 120  115   8030  "Super Large"
_add "Qwen/Qwen3.5-122B-A10B"                                    "Qwen3.5-122B-A10B"                     "Qwen3.5-122B-A10B (BF16) [SUPER]"           122  120   8031  "Super Large"
_add "Qwen/Qwen3.5-122B-A10B-FP8"                               "Qwen3.5-122B-A10B-FP8"                 "Qwen3.5-122B-A10B (FP8) [SUPER] ★Rec"        62   65   8033  "Super Large"
_add "openai/gpt-oss-120b"                                       "gpt-oss-120b"                          "GPT-OSS-120B [SUPER]"                       120  115   8032  "Super Large"

# NVFP4 resharded specifically for DGX Spark's 128GB unified pool: same weights
# as RedHatAI's Qwen3.5-122B-A10B NVFP4 quantization, just repacked from 2 large
# shards into 16 x ~4.7GB shards to avoid memory-allocation failures on load.
# https://huggingface.co/sjug/Qwen3.5-122B-A10B-NVFP4-resharded
# ~71 GB disk / ~72 GB actual VRAM (smallest of the three 122B-A10B profiles
# here, and the only one that supports the full 262144 context — see the
# _serve_model entry below for why).
_add "sjug/Qwen3.5-122B-A10B-NVFP4-resharded"                    "Qwen3.5-122B-A10B-NVFP4-spark"         "Qwen3.5-122B-A10B-NVFP4 (Spark resharded) [SUPER]" 71 72   8034  "Super Large"

# ── Small models with a custom idle-sleep timeout ─────────────────────────────
# These pass the optional 8th _add field (SLEEP_MIN) = 60, so the sleep watchdog
# offloads them to CPU after 1 hour idle instead of the global IDLE_SLEEP_MINUTES.
# Appended at the end of the catalog so the existing indices above (and their
# matching serve blocks) are not shifted.
#        HF Repo                       Local Dir              Display Name                              Disk VRAM  Port  Category    Sleep(min)
_add "Qwen/Qwen3.5-4B"               "Qwen3.5-4B"           "Qwen3.5-4B (BF16) [1h sleep]"            8   10   8012  "Dense Models"   60
_add "Qwen/Qwen3.5-2B"               "Qwen3.5-2B"           "Qwen3.5-2B (BF16) [1h sleep]"            4    5   8013  "Dense Models"   60
_add "Qwen/Qwen3.5-9B"               "Qwen3.5-9B"           "Qwen3.5-9B (BF16) [1h sleep]"           18   18   8015  "Dense Models"   60

# ── Additional ASR / NeMo model (download-only, not served via vLLM) ───────────
# https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
# Appended here (not next to idx 11/12) to keep existing catalog indices stable.
_add "nvidia/nemotron-3.5-asr-streaming-0.6b" "nemotron-3.5-asr-streaming-0.6b" "Nemotron-3.5-ASR-Streaming-0.6B (ASR)"   1    0      0  "ASR"

# ── ASR model served via vLLM (transcription endpoint, gets a port) ───────────
# https://huggingface.co/Qwen/Qwen3-ASR-1.7B
# Unlike the NeMo ASR models above, this is served by vLLM (--task transcription)
# and exposes POST /v1/audio/transcriptions. PORT is non-zero so it shows up in
# the serve menu and can be selected like any other model.
_add "Qwen/Qwen3-ASR-1.7B"           "Qwen3-ASR-1.7B"       "Qwen3-ASR-1.7B (ASR, served)"            4    5   8014  "ASR"

# ── Qwen3.6 NVFP4 quantizations ───────────────────────────────────────────────
# NVFP4 is ~4-bit. The 27B dense checkpoint is sized to co-run with the 35B-A3B
# MoE at ITS old, smaller profile — that no longer holds since v0.3.26 moved
# 35B-A3B to the full 262144-context profile below (0.4 gmu, fp8 KV cache), so
# don't launch both together without checking free memory first.
# https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
# https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
# https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4
#        HF Repo                                Local Dir                       Display Name                              Disk VRAM  Port  Category
_add "nvidia/Qwen3.6-35B-A3B-NVFP4"          "Qwen3.6-35B-A3B-NVFP4"        "Qwen3.6-35B-A3B (NVFP4, nvidia)"         20   49   8016  "MoE Models"
_add "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"    "Qwen3.6-35B-A3B-NVFP4-Fast"   "Qwen3.6-35B-A3B (NVFP4-Fast, unsloth)"   20   22   8017  "MoE Models"
_add "nvidia/Qwen3.6-27B-NVFP4"               "Qwen3.6-27B-NVFP4"            "Qwen3.6-27B (NVFP4, nvidia)"             18   24   8022  "Dense Models"

# ── Nemotron-3-Nano-Omni-30B-A3B-Reasoning quantizations (added v0.3.6) ────────
# Quantized builds of the BF16 Omni reasoning model above (idx served on 8008).
# FP8 ≈ half the BF16 footprint; NVFP4 (~4-bit) ≈ a quarter. Both are multimodal
# ("Omni") reasoning models, served with --trust-remote-code like the BF16 build.
# https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8
# https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
#        HF Repo                                                     Local Dir                                       Display Name                          Disk VRAM  Port  Category
_add "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"          "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"    "Nemotron-3-Nano-Omni-30B (FP8)"      30   34   8018  "MoE Models"
_add "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"  "Nemotron-3-Nano-Omni-30B (NVFP4)"    18   20   8019  "MoE Models"

# ── Gemma 4 31B IT NVFP4 quantization ─────────────────────────────────────────
# https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
# vLLM/ModelOpt NVFP4 profile for Blackwell. This text-only launch profile uses
# 0.46 gpu-memory-utilization (~56 GB on DGX Spark) and 32768 context to reduce
# memory pressure versus the 65536 / 0.82 profile.
#        HF Repo                                Local Dir                       Display Name                              Disk VRAM  Port  Category
_add "nvidia/Gemma-4-31B-IT-NVFP4"            "Gemma-4-31B-IT-NVFP4"         "Gemma 4 31B IT (NVFP4, nvidia)"          24   56   8001  "Dense Models"

MODEL_TOTAL=${#MDL_HF[@]}

# ── Default pre-selected models ────────────────────────────────────────────────
# Nothing is pre-selected — the user picks models in the interactive menus (or
# names them with --start). Add catalog indices here to pre-check them.
DEFAULT_DL_INDICES=()
DEFAULT_SERVE_INDICES=()

# ─────────────────────────────────────────────────────────────────────────────
# HEADLESS HELPERS — model resolution, catalog listing, @reboot cron management
# ─────────────────────────────────────────────────────────────────────────────

# Resolve a user-supplied model reference to a catalog index (echoed on stdout).
# Accepts: catalog index, local dir name, full HF repo id, or the repo basename.
# Matching is case-insensitive. Returns 1 if not found or not servable (PORT=0).
_resolve_model() {
    local q="$1" i lq dir_lc hf_lc
    if [[ "$q" =~ ^[0-9]+$ ]]; then
        [ "$q" -lt "$MODEL_TOTAL" ] && [ "${MDL_PORT[$q]}" != "0" ] && { echo "$q"; return 0; }
        return 1
    fi
    lq="$(printf '%s' "$q" | tr '[:upper:]' '[:lower:]')"
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "${MDL_PORT[$i]}" = "0" ] && continue
        dir_lc="$(printf '%s' "${MDL_DIR[$i]}" | tr '[:upper:]' '[:lower:]')"
        hf_lc="$(printf '%s' "${MDL_HF[$i]}"  | tr '[:upper:]' '[:lower:]')"
        if [ "$lq" = "$dir_lc" ] || [ "$lq" = "$hf_lc" ] || [ "$lq" = "${hf_lc##*/}" ]; then
            echo "$i"; return 0
        fi
    done
    return 1
}

# Print every servable model (PORT != 0) — the names accepted by --start.
_list_servable_models() {
    echo ""
    printf "  %-4s  %-42s  %-52s  %5s  %6s\n" "Idx" "Name for --start (local dir)" "HF repo" "Port" "VRAM"
    printf "  %-4s  %-42s  %-52s  %5s  %6s\n" "---" "----------------------------" "-------" "----" "----"
    local i
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "${MDL_PORT[$i]}" = "0" ] && continue
        printf "  %-4d  %-42s  %-52s  %5s  %3d GB\n" \
            "$i" "${MDL_DIR[$i]}" "${MDL_HF[$i]}" "${MDL_PORT[$i]}" "${MDL_VRAM[$i]}"
    done
    echo ""
}

# Parse a "model[:port],model[:port],..." spec into RUN_SELECTED, applying any
# per-model port overrides directly to MDL_PORT so logs / watchdog / status all
# see the overridden port. Exits with the catalog listed on any bad entry.
_resolve_start_specs() {
    local spec="$1" part m p idx
    local -a parts
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [ -z "$part" ] && continue
        m="${part%%:*}"; p=""
        [[ "$part" == *:* ]] && p="${part##*:}"
        if ! idx=$(_resolve_model "$m"); then
            echo "❌ Unknown or non-servable model: '$m'"
            echo "   Use one of the names below (or a catalog index):"
            _list_servable_models
            exit 1
        fi
        if [ -n "$p" ]; then
            [[ "$p" =~ ^[0-9]+$ ]] || { echo "❌ Invalid port '$p' for model '$m'"; exit 1; }
            MDL_PORT[$idx]="$p"
            MDL_PORT_EXPLICIT[$idx]=1   # pinned — exempt from sequential re-assignment
        fi
        RUN_SELECTED+=("$idx")
    done
    [ "${#RUN_SELECTED[@]}" -eq 0 ] && { echo "❌ --start: no models resolved from '$spec'"; exit 1; }
}

# Reassign servable models to sequential ports (BASE_PORT, +1, +2, …) in launch
# order, so the same selection always maps to the same ports. Models pinned via
# --start "model:PORT" keep their port and are skipped over (their port is also
# avoided so sequential assignment never collides with a pin). Download-only
# models (PORT=0) are left untouched.
_assign_sequential_ports() {
    local idx next="$BASE_PORT" pinned=" "
    # Collect pinned ports first so sequential assignment can skip past them.
    for idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT_EXPLICIT[$idx]:-0}" = "1" ] && pinned="${pinned}${MDL_PORT[$idx]} "
    done
    for idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$idx]}" = "0" ] && continue                 # download-only
        [ "${MDL_PORT_EXPLICIT[$idx]:-0}" = "1" ] && continue      # user-pinned
        while [[ "$pinned" == *" $next "* ]]; do next=$((next + 1)); done
        MDL_PORT[$idx]="$next"
        next=$((next + 1))
    done
}

# Validate a "model[:port],..." spec against the catalog — same rules --start
# applies via _resolve_start_specs, but without mutating MDL_PORT/RUN_SELECTED
# (this just needs to know the spec is launchable before it gets persisted).
# Exits with the catalog listed on any bad entry — never persists a broken spec.
_validate_spec_or_die() {
    local spec="$1"
    local -a parts; local part m p
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [ -z "$part" ] && continue
        m="${part%%:*}"; p=""
        [[ "$part" == *:* ]] && p="${part##*:}"
        _resolve_model "$m" >/dev/null || { echo "❌ Unknown model '$m' in spec:"; _list_servable_models; exit 1; }
        [ -n "$p" ] && ! [[ "$p" =~ ^[0-9]+$ ]] && { echo "❌ Invalid port '$p' for '$m'"; exit 1; }
    done
}

# Persist the boot-time spec read by --start-saved. Decoupled from the crontab
# entry itself — this can be called any time (even repeatedly) without ever
# touching cron, so "which model boots" is a one-line file write, not a cron edit.
_set_boot_model() {
    local spec="$1"
    _validate_spec_or_die "$spec"
    mkdir -p "$BASE_DIR"
    printf '%s' "$spec" > "$BOOT_MODEL_SPEC_FILE"
    echo "✅ Boot model saved: $spec"
    if crontab -l 2>/dev/null | grep -qE "$(basename "$0")' --start-saved"; then
        echo "   Takes effect on next reboot (or run now: $0 --start-saved)"
    else
        echo "   ⚠️  No @reboot cron entry yet — run this once to enable auto-start at boot:"
        echo "        $0 --install-cron"
    fi
}

# Install/replace the @reboot crontab entry that re-launches the saved boot
# model at boot. The line is STABLE — it always runs `--start-saved`, so
# changing which model boots is a --set-boot-model call, never a crontab edit.
# The 60s sleep gives the network / GPU driver time to come up; adjust if needed.
_install_boot_cron() {
    local spec="$1"
    if [ -n "$spec" ]; then
        _set_boot_model "$spec"   # also validates + persists
    elif [ ! -s "$BOOT_MODEL_SPEC_FILE" ]; then
        echo "❌ --install-cron: no spec given and no saved boot model yet."
        echo "   Either: $0 --install-cron '<model[:port][,model...]>'"
        echo "   Or:     $0 --set-boot-model '<spec>'   (then --install-cron with no args)"
        exit 1
    fi

    local script_path="$SCRIPT_DIR/$(basename "$0")"
    local cron_line="@reboot sleep 60 && mkdir -p '$BASE_DIR/logs' && '$script_path' --start-saved >> '$BASE_DIR/logs/startup_vllm.log' 2>&1"
    ( crontab -l 2>/dev/null | grep -vE "$(basename "$0")' --start" ; echo "$cron_line" ) | crontab -
    echo "✅ @reboot cron entry installed (boot model: $(cat "$BOOT_MODEL_SPEC_FILE" 2>/dev/null)):"
    echo "   $cron_line"
    echo "   Change the boot model any time — no crontab edit needed:"
    echo "     $0 --set-boot-model '<model[:port][,model...]>'"
    echo "   Boot log: $BASE_DIR/logs/startup_vllm.log"
    echo "   Remove with: $0 --remove-cron"
}

_remove_boot_cron() {
    if crontab -l 2>/dev/null | grep -qE "$(basename "$0")' --start"; then
        crontab -l 2>/dev/null | grep -vE "$(basename "$0")' --start" | crontab -
        echo "✅ @reboot vLLM startup cron entry removed."
    else
        echo "ℹ️  No @reboot vLLM startup cron entry found — nothing to remove."
    fi
}

# ── One-shot actions: list catalog / manage cron / manage boot model, then exit
if [ "$LIST_MODELS" -eq 1 ]; then
    _list_servable_models
    exit 0
fi
if [ -n "$SET_BOOT_MODEL_SPEC" ]; then
    _set_boot_model "$SET_BOOT_MODEL_SPEC"; exit 0
fi
if [ "$CRON_ACTION" = "install" ]; then
    _install_boot_cron "$CRON_SPEC"; exit 0
elif [ "$CRON_ACTION" = "remove" ]; then
    _remove_boot_cron; exit 0
fi

# --start-saved: not one-shot — resolves the persisted boot spec into
# START_SPECS and falls through into the normal headless serve flow below,
# exactly as if the user had typed --start '<saved spec>'.
if [ "$START_SAVED" -eq 1 ]; then
    START_SPECS="$(cat "$BOOT_MODEL_SPEC_FILE" 2>/dev/null || true)"
    if [ -z "$START_SPECS" ]; then
        echo "❌ --start-saved: no boot model saved yet. Set one with:"
        echo "   $0 --set-boot-model '<model[:port][,model...]>'"
        exit 1
    fi
    HEADLESS=1
    SERVE_ONLY=1
    echo "ℹ️  --start-saved: using saved boot model spec: $START_SPECS"
fi

# Ports worth probing for a live model: every catalog port PLUS the sequential
# BASE_PORT block (served ports are assigned BASE_PORT, +1, …). Deduped/sorted.
# Covers both catalog-port and sequential runs, and works in a standalone
# invocation where MDL_PORT still holds catalog defaults.
_serve_probe_ports() {
    local i out="" n=0
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "${MDL_PORT[$i]}" = "0" ] && continue
        out="$out ${MDL_PORT[$i]}"
        n=$((n + 1))
    done
    for i in $(seq 0 $((n + 3))); do out="$out $((BASE_PORT + i))"; done
    printf '%s\n' $out | sort -un
}

# ─────────────────────────────────────────────────────────────────────────────
# STATUS SNAPSHOT — memory utilization + models currently live in vLLM
# Probes catalog ports and the sequential BASE_PORT block. Called once at
# startup (shows what a previous run left running) and once at the end.
# Arg: $1 = label for the section header (e.g. "STARTUP", "FINAL").
# ─────────────────────────────────────────────────────────────────────────────
_show_vllm_status() {
    local label="$1"
    echo ""
    echo "  ══ vLLM / MEMORY STATUS — ${label} ══════════════════════════════"

    # ---- System RAM (used / total) ----
    local mt ma
    mt=$(awk '/^MemTotal:/{print $2}'     /proc/meminfo 2>/dev/null)
    ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)
    if [[ "$mt" =~ ^[0-9]+$ ]] && [[ "$ma" =~ ^[0-9]+$ ]]; then
        printf "  RAM     : %d / %d GB used  (%d%%)\n" \
            "$(( (mt - ma) / 1024 / 1024 ))" "$(( mt / 1024 / 1024 ))" "$(( (mt - ma) * 100 / mt ))"
    else
        echo "  RAM     : /proc/meminfo unavailable"
    fi

    # ---- GPU (best effort; unified-memory systems report N/A for memory) ----
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        local gutil gmu gmt
        gutil=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        gmu=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        gmt=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        [[ "$gutil" =~ ^[0-9]+$ ]] && printf "  GPU     : %d%% compute utilization\n" "$gutil"
        if [[ "$gmu" =~ ^[0-9]+$ ]] && [[ "$gmt" =~ ^[0-9]+$ ]]; then
            printf "  GPU mem : %d / %d MiB used\n" "$gmu" "$gmt"
        else
            echo "  GPU mem : shared with system RAM (nvidia-smi reports N/A)"
        fi
    else
        echo "  GPU     : nvidia-smi not available"
    fi

    # ---- Models currently live in vLLM (probe catalog + sequential ports) ----
    echo "  Models live in vLLM:"
    local found=0 p
    for p in $(_serve_probe_ports); do
        local resp
        resp=$(curl -sf --max-time 2 "http://localhost:${p}/v1/models" 2>/dev/null) || continue
        local ids
        if command -v jq &>/dev/null; then
            ids=$(echo "$resp" | jq -r '.data[].id' 2>/dev/null | paste -sd ',' -)
        else
            ids=$(echo "$resp" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
                  | sed -E 's/.*"([^"]*)"$/\1/' | paste -sd ',' -)
        fi
        [ -z "$ids" ] && ids="(up — model id unknown)"
        printf "    port %-6s : %s\n" "$p" "$ids"
        found=1
    done
    [ "$found" = "0" ] && echo "    (none responding)"
    echo "  ════════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL HEALTH CHECK — confirm each served model actually answers on its port.
# For every model in RUN_SELECTED it curls /health (is the engine up?) and
# /v1/models (which model id is registered?), printing UP or NOT READY plus the
# exact commands to re-test and to tail that model's log. Returns 0 only if all
# selected servable models are up, so callers can branch on the result.
# ─────────────────────────────────────────────────────────────────────────────
_verify_served_models() {
    local all_up=1 any=0 idx port name log_file resp served_id
    echo ""
    echo "  ══ MODEL HEALTH CHECK ═══════════════════════════════════════════"
    for idx in "${RUN_SELECTED[@]}"; do
        port="${MDL_PORT[$idx]}"
        [ "$port" = "0" ] && continue
        any=1
        name="${MDL_NAME[$idx]}"
        log_file="$VLLM_LOGS/vllm-${port}.log"
        if curl -sf --max-time 5 "http://localhost:${port}/health" > /dev/null 2>&1; then
            resp=$(curl -sf --max-time 5 "http://localhost:${port}/v1/models" 2>/dev/null)
            if command -v jq &>/dev/null; then
                served_id=$(echo "$resp" | jq -r '.data[0].id // empty' 2>/dev/null)
            else
                served_id=$(echo "$resp" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
                            | head -1 | sed -E 's/.*"([^"]*)"$/\1/')
            fi
            printf "  ✅ UP         %-42s (port %s)\n" "$name" "$port"
            [ -n "$served_id" ] && printf "                served id : %s\n" "$served_id"
            printf "                test      : curl -s http://localhost:%s/v1/models | jq .\n" "$port"
        else
            all_up=0
            printf "  ⏳ NOT READY  %-42s (port %s)\n" "$name" "$port"
            printf "                still loading or failed to start — watch the log:\n"
            printf "                tail -f %s\n" "$log_file"
            printf "                re-check  : curl -s http://localhost:%s/health\n" "$port"
        fi
    done
    [ "$any" = "0" ] && echo "  (no servable models were selected)"
    echo "  ═════════════════════════════════════════════════════════════════"
    if [ "$any" = "1" ]; then
        if [ "$all_up" = "1" ]; then
            echo "  ✅ All selected models are UP and answering."
        else
            echo "  ⏳ Some models are not answering yet. vLLM takes ~5-10 min to load a"
            echo "     large model; re-run this health check any time with:"
            echo "         $0 --health"
        fi
    fi
    [ "$all_up" = "1" ] && return 0 || return 1
}

# ── One-shot: --health probes the live ports and reports what's up ───────────
# Standalone re-check (e.g. minutes after launch, or from cron/monitoring). A
# fresh invocation can't know the launch order, so it probes the catalog ports
# and the sequential BASE_PORT block and lists only the ports that answer,
# reading each served model id straight from /v1/models.
if [ "$HEALTH_CHECK" -eq 1 ]; then
    VLLM_LOGS="$BASE_DIR/logs"
    echo ""
    echo "  ══ MODEL HEALTH CHECK ═══════════════════════════════════════════"
    _hc_found=0
    for _p in $(_serve_probe_ports); do
        curl -sf --max-time 2 "http://localhost:${_p}/health" > /dev/null 2>&1 || continue
        _hc_found=1
        _resp=$(curl -sf --max-time 3 "http://localhost:${_p}/v1/models" 2>/dev/null)
        if command -v jq &>/dev/null; then
            _id=$(echo "$_resp" | jq -r '.data[0].id // empty' 2>/dev/null)
        else
            _id=$(echo "$_resp" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
                  | head -1 | sed -E 's/.*"([^"]*)"$/\1/')
        fi
        printf "  ✅ UP    port %-6s  %s\n" "$_p" "${_id:-(model id unknown)}"
        printf "           test : curl -s http://localhost:%s/v1/models | jq .\n" "$_p"
        [ -f "$VLLM_LOGS/vllm-${_p}.log" ] && \
            printf "           log  : tail -f %s/vllm-%s.log\n" "$VLLM_LOGS" "$_p"
    done
    echo "  ═════════════════════════════════════════════════════════════════"
    if [ "$_hc_found" = "0" ]; then
        echo "  ⚠️  No vLLM models are currently answering (checked catalog ports and"
        echo "     the ${BASE_PORT}+ sequential block)."
        echo "     If you just started one, give it ~5-10 min and re-run: $0 --health"
        echo "     Startup logs: $VLLM_LOGS/vllm-<port>.log"
        exit 1
    fi
    exit 0
fi

# Startup snapshot — what a previous run left running, before the clean-start kill.
_show_vllm_status "STARTUP"

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CHECKBOX SELECTION
# ─────────────────────────────────────────────────────────────────────────────

_checkbox_menu() {
    # Args: $1=title  $2=servable_only (true|false)  $3=result_var_name  $4=defaults_var (optional)
    local title="$1" servable_only="$2" result_var="$3" defaults_var="${4:-}"

    local -a menu_map=()
    for i in $(seq 0 $((MODEL_TOTAL - 1))); do
        [ "$servable_only" = "true" ] && [ "${MDL_PORT[$i]}" = "0" ] && continue
        menu_map+=("$i")
    done

    # ── Display order: group by type, then size (VRAM, ascending), then name ───
    # COSMETIC ONLY. This reorders what the menu shows without changing catalog
    # indices, so every serve block, DEFAULT_*_INDICES, and STANDARD_INDICES keep
    # working unchanged. (Physically reordering the _add lines would shift indices
    # and mis-map serve blocks — the v0.1.5 bug.)
    if [ "${#menu_map[@]}" -gt 1 ]; then
        local -a _sortable=()
        local _mi _crank
        for _mi in "${menu_map[@]}"; do
            case "${MDL_CAT[$_mi]}" in
                "MoE Models")   _crank=1 ;;
                "Dense Models") _crank=2 ;;
                Embeddings)     _crank=3 ;;
                Reranking)      _crank=4 ;;
                ASR)            _crank=5 ;;
                "Super Large")  _crank=6 ;;
                *)              _crank=9 ;;
            esac
            _sortable+=("$(printf '%d|%04d|%s|%d' "$_crank" "${MDL_VRAM[$_mi]}" "${MDL_NAME[$_mi]}" "$_mi")")
        done
        menu_map=()
        while IFS='|' read -r _ _ _ _mi; do menu_map+=("$_mi"); done \
            < <(printf '%s\n' "${_sortable[@]}" | sort -t'|' -k1,1n -k2,2n -k3,3)
    fi

    local count=${#menu_map[@]}

    local -a sel=()
    local -a sel_order=()
    for j in $(seq 0 $((count - 1))); do sel[$j]=0; done

    # Pre-select any default catalog indices passed via defaults_var
    if [ -n "$defaults_var" ]; then
        local -n _defs_ref="$defaults_var"
        for def_idx in "${_defs_ref[@]+${_defs_ref[@]}}"; do
            for j in $(seq 0 $((count - 1))); do
                if [ "${menu_map[$j]}" = "$def_idx" ]; then
                    sel[$j]=1
                    sel_order+=("$j")
                fi
            done
        done
    fi

    while true; do
        echo ""
        echo "  $title"
        printf "  %-4s  %-3s  %-50s  %7s  %10s  %-12s\n" \
               "Num" " " "Model" "Disk" "VRAM" "Category"
        printf "  %-4s  %-3s  %-50s  %7s  %10s  %-12s\n" \
               "---" "---" "-----" "------" "----------" "--------"

        for j in $(seq 0 $((count - 1))); do
            local i="${menu_map[$j]}"
            local mark="[ ]"; [ "${sel[$j]}" = "1" ] && mark="[x]"
            local vram_disp="CPU"
            [ "${MDL_VRAM[$i]}" -gt 0 ] && vram_disp="${MDL_VRAM[$i]} GB"
            printf "  %-4d  %s  %-50s  %4d GB  %10s  %-12s\n" \
                "$((j+1))" "$mark" "${MDL_NAME[$i]}" "${MDL_DISK[$i]}" "$vram_disp" "${MDL_CAT[$i]}"
        done

        local selected_count=0
        for j in $(seq 0 $((count - 1))); do [ "${sel[$j]}" = "1" ] && selected_count=$((selected_count+1)); done
        echo ""
        printf "  [%d selected]  Toggle: type number(s) separated by spaces\n" "$selected_count"
        echo "  Commands: a=select all  n=clear all  d=done"
        printf "  > "
        read -r input

        case "$input" in
            d|done|"") break ;;
            a|all)
                sel_order=()
                for j in $(seq 0 $((count - 1))); do
                    sel[$j]=1
                    sel_order+=("$j")
                done ;;
            n|none|clear)
                for j in $(seq 0 $((count - 1))); do sel[$j]=0; done
                sel_order=() ;;
            *)
                for num in $input; do
                    if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "$count" ]; then
                        local j=$((num - 1))
                        if [ "${sel[$j]}" = "1" ]; then
                            sel[$j]=0
                            local -a _new_order=()
                            local _oj
                            for _oj in "${sel_order[@]+${sel_order[@]}}"; do
                                [ "$_oj" != "$j" ] && _new_order+=("$_oj")
                            done
                            sel_order=("${_new_order[@]+${_new_order[@]}}")
                        else
                            sel[$j]=1
                            sel_order+=("$j")
                        fi
                    fi
                done ;;
        esac
    done

    local -a result=()
    local seen_order=" "
    for j in "${sel_order[@]+${sel_order[@]}}"; do
        if [ "${sel[$j]:-0}" = "1" ]; then
            result+=("${menu_map[$j]}")
            seen_order="${seen_order}${j} "
        fi
    done
    for j in $(seq 0 $((count - 1))); do
        [ "${sel[$j]}" = "1" ] || continue
        [[ "$seen_order" == *" $j "* ]] && continue
        result+=("${menu_map[$j]}")
    done
    eval "$result_var=(\"\${result[@]+\"\${result[@]}\"}\") "
}

# ─────────────────────────────────────────────────────────────────────────────
# VRAM PRE-FLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────────────

# Budget required model memory against system RAM (for unified-memory systems
# like the DGX Spark / GB10, where the GPU and CPU share one memory pool) and
# do a live memory-pressure check against what is actually free right now.
# Arg: $1 = total GB the selected models require.
_check_system_ram_budget() {
    local total_required="$1"
    local mem_total_kb mem_avail_kb
    mem_total_kb=$(awk '/^MemTotal:/{print $2}'     /proc/meminfo 2>/dev/null)
    mem_avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)

    if ! [[ "$mem_total_kb" =~ ^[0-9]+$ ]]; then
        echo "  ⚠️  Could not read /proc/meminfo — skipping memory check."
        echo "     Selected models require ~${total_required} GB; ensure your RAM fits."
        return 0
    fi

    local total_gb=$(( mem_total_kb / 1024 / 1024 ))
    local avail_gb=0
    [[ "$mem_avail_kb" =~ ^[0-9]+$ ]] && avail_gb=$(( mem_avail_kb / 1024 / 1024 ))
    # 85% safe limit leaves headroom for the OS, KV cache, and the OpenWebUI /
    # Qdrant / SearXNG containers that share the same RAM.
    local safe_gb=$(( total_gb * 85 / 100 ))

    local used_gb=$(( total_gb - avail_gb ))
    printf "  %-20s : %3d GB   total physical RAM on this box\n"               "System RAM total" "$total_gb"
    printf "  %-20s : %3d GB   in use right now (OS, containers, old models)\n" "In use now"        "$used_gb"
    printf "  %-20s : %3d GB   free right now\n"                               "Available now"     "$avail_gb"
    printf "  %-20s : %3d GB   most you should load (85%% of total)\n"          "Safe limit"        "$safe_gb"
    printf "  %-20s : %3d GB   sum of selected models' VRAM estimates\n"        "Models require"    "$total_required"
    echo ""

    # Two distinct conditions, with different meanings and advice:
    #   capacity  — models exceed the safe limit; they won't fit even when idle.
    #   transient — models fit overall, but not enough is free this instant
    #               (usually a previous run's models, killed later in this script).
    local pressure="" kind=""
    if [ "$total_required" -gt "$safe_gb" ]; then
        kind="capacity"
        pressure="Models need ~${total_required} GB, over the ${safe_gb} GB safe limit (85% of ${total_gb} GB)."
    elif [ "$avail_gb" -gt 0 ] && [ "$total_required" -gt "$avail_gb" ]; then
        kind="transient"
        pressure="Models need ~${total_required} GB but only ${avail_gb} GB is free this instant (${used_gb} GB already in use)."
    fi

    if [ -n "$pressure" ]; then
        echo "  ⚠️  MEMORY PRESSURE: $pressure"
        if [ "$kind" = "transient" ]; then
            echo "     This is almost always because a PREVIOUS run's models are still"
            echo "     loaded (see the 'STARTUP' snapshot above). This script kills old"
            echo "     vLLM processes a few steps from now, which frees that RAM — so since"
            echo "     ${total_required} GB is under the ${safe_gb} GB safe limit, you can most"
            echo "     likely continue. If models then fail, watch their logs for OOM errors."
        else
            echo "     ${total_required} GB won't fit safely even on an idle box. KV cache,"
            echo "     OpenWebUI, Qdrant, and the OS also draw from this same shared pool."
            echo "     Deselect some models, or pick smaller / quantized (FP8/NVFP4) variants."
        fi
        echo -n "  Continue anyway? [y/N]: "
        read -r confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    else
        echo "  ✅ Memory check OK — need ~${total_required} GB; ${avail_gb} GB free now,"
        echo "     ${safe_gb} GB safe limit, ${total_gb} GB total."
    fi
}

# Kill any running vLLM model processes (a previous run's, or this script's).
# Shared by the optional pre-check shutdown and the later clean-start.
_kill_vllm_processes() {
    pkill -9 -f "vllm serve"        2>/dev/null || true
    pkill -9 -f "vllm.entrypoints"  2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore"  2>/dev/null || true
    pkill -9 -f "vllm.engine"       2>/dev/null || true
    # Also stop a previous run's sleep watchdog so watchdogs don't stack.
    pkill -f "sleep_watchdog.sh"    2>/dev/null || true
}

# Echo the PID(s) of whatever is LISTENING on a TCP port (space-separated, empty
# if the port is free). Tries lsof, then ss, then fuser — whichever exists.
_port_listener_pids() {
    local port="$1" pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)
    elif command -v ss >/dev/null 2>&1; then
        pids=$(ss -ltnpH "( sport = :$port )" 2>/dev/null \
               | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
    elif command -v fuser >/dev/null 2>&1; then
        pids=$(fuser "${port}/tcp" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$')
    fi
    echo $pids   # unquoted: normalize newlines/space to single spaces
}

# Make sure PORT is free before a model binds it. If something is already
# listening, show what it is and:
#   • interactive → ask whether to kill it (default No; No/failed-kill = skip model)
#   • headless    → reclaim only if it looks like a prior vLLM process (never kill
#                   unrelated services); otherwise skip this model.
# Returns 0 if the port is free (or was freed), 1 if the caller should skip.
_ensure_port_available() {
    local port="$1" name="$2" pids pid line
    pids=$(_port_listener_pids "$port")
    [ -z "$pids" ] && return 0

    echo ""
    echo "  ⚠️  Port $port (needed for $name) is already in use by:"
    for pid in $pids; do
        line=$(ps -p "$pid" -o pid=,comm=,args= 2>/dev/null)
        [ -n "$line" ] && echo "        $line" || echo "        pid $pid (details unavailable)"
    done

    if [ "$HEADLESS" -eq 1 ]; then
        if ps -p "${pids// /,}" -o args= 2>/dev/null | grep -qiE 'vllm|api_server'; then
            echo "  → Headless: looks like a previous vLLM instance — reclaiming port (kill $pids)."
            kill -9 $pids 2>/dev/null || true
            sleep 2
            [ -z "$(_port_listener_pids "$port")" ] && { echo "  ✅ Port $port freed."; return 0; }
            echo "  ⚠️  Port $port still busy after kill — skipping $name."; return 1
        fi
        echo "  → Headless (no prompt): not a vLLM process — leaving it and SKIPPING $name."
        return 1
    fi

    printf "  Kill the process(es) on port %s and continue? [y/N]: " "$port"
    read -r _ans
    if [[ "$_ans" =~ ^[Yy]$ ]]; then
        kill -9 $pids 2>/dev/null || true
        sleep 2
        if [ -z "$(_port_listener_pids "$port")" ]; then
            echo "  ✅ Port $port freed."
            return 0
        fi
        echo "  ⚠️  Port $port still in use after kill — skipping $name."
        return 1
    fi
    echo "  → Leaving it; SKIPPING $name (port busy)."
    return 1
}

# Ask whether to shut down a previous run's still-loaded models before the memory
# check, so "Available now" reflects a clean slate instead of stale reservations.
# Only prompts when vLLM processes are actually detected.
_maybe_shutdown_existing_models() {
    pgrep -f "vllm serve" >/dev/null 2>&1 || pgrep -f "vllm.entrypoints" >/dev/null 2>&1 || return 0

    echo ""
    echo "  ⚠️  vLLM model(s) from a previous run are still loaded and holding memory"
    echo "      (see the STARTUP snapshot above). Shutting them down now frees that"
    echo "      memory so the budget check below reflects a clean slate."
    echo -n "  Shut down existing models now? [Y/n]: "
    read -r _kill_ans
    if [[ "$_kill_ans" =~ ^[Nn]$ ]]; then
        echo "  → Keeping existing models; 'Available now' will still include them."
    else
        echo "  → Stopping existing vLLM processes..."
        _kill_vllm_processes
        sleep 3
        echo "  ✅ Existing models stopped — memory freed."
    fi
}

_check_vram() {
    local total_required=0
    for idx in "${RUN_SELECTED[@]}"; do
        total_required=$((total_required + MDL_VRAM[idx]))
    done
    [ "$total_required" -eq 0 ] && return 0

    echo ""
    echo "  ── VRAM Budget ──────────────────────────────────────────"

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        local total_mib
        total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)

        # Unified-memory systems (e.g. DGX Spark / GB10) report memory.total as
        # "[N/A]". The GPU shares system RAM, so budget against that instead.
        if ! [[ "$total_mib" =~ ^[0-9]+$ ]]; then
            echo "  ℹ️  Unified memory detected (nvidia-smi VRAM = '${total_mib:-N/A}')."
            echo "     GPU and CPU share one pool — budgeting against system RAM."
            _check_system_ram_budget "$total_required"
            echo "  ─────────────────────────────────────────────────────────"
            return 0
        fi

        local total_gb=$(( total_mib / 1024 ))
        local safe_gb=$(( total_gb * 90 / 100 ))

        printf "  %-20s : %d GB\n" "GPU total" "$total_gb"
        printf "  %-20s : %d GB  (90%%)\n" "Safe limit" "$safe_gb"
        printf "  %-20s : %d GB\n" "Models require" "$total_required"
        echo ""

        local has_super=0
        for idx in "${RUN_SELECTED[@]}"; do
            [ "${MDL_CAT[$idx]}" = "Super Large" ] && has_super=1 && break
        done

        if [ "$has_super" = "1" ] && [ "${#RUN_SELECTED[@]}" -gt 1 ]; then
            echo "  ⚠️  WARNING: You selected a SUPER LARGE model alongside other models."
            echo "     Super Large models (120B+) need nearly all GPU VRAM."
            echo "     Running multiple large models simultaneously will likely fail."
            echo -n "  Continue anyway? [y/N]: "
            read -r confirm
            [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
        elif [ "$total_required" -gt "$safe_gb" ]; then
            echo "  ⚠️  WARNING: Selected models require ~${total_required} GB but safe limit is ${safe_gb} GB."
            echo "     Consider deselecting some models."
            echo -n "  Continue anyway? [y/N]: "
            read -r confirm
            [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
        else
            echo "  ✅ VRAM check OK — ${total_required} GB needed, ${total_gb} GB available"
        fi
    else
        echo "  ⚠️  nvidia-smi not available — budgeting against system RAM instead."
        _check_system_ram_budget "$total_required"
    fi
    echo "  ─────────────────────────────────────────────────────────"
}

# ─────────────────────────────────────────────────────────────────────────────
# vLLM AUTO-UPDATE — upgrade the venv's vLLM to the latest PyPI release if newer.
# Best-effort: never fails the run. Called once VENV_DIR is known (all modes).
# ─────────────────────────────────────────────────────────────────────────────
_maybe_update_vllm() {
    [ "${AUTO_UPDATE_VLLM:-true}" = "true" ] || return 0
    local py="$VENV_DIR/bin/python" pip="$VENV_DIR/bin/pip"
    if [ ! -x "$py" ] || [ ! -x "$pip" ]; then
        echo "ℹ️  vLLM auto-update skipped — venv python/pip not found under $VENV_DIR"
        return 0
    fi

    echo "--- Checking for a newer vLLM (AUTO_UPDATE_VLLM=true) ---"
    local cur
    cur=$("$py" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
    if [ -z "$cur" ]; then
        echo "⚠️  vllm not importable in the venv — installing the latest release…"
        "$pip" install -U vllm 2>&1 | tail -4
        "$py" -c 'import vllm; print("✅ vllm installed:", vllm.__version__)' 2>/dev/null \
            || echo "❌ vllm still not importable — check the pip output above."
        _fix_flashinfer_versions "$VENV_DIR"
        return 0
    fi

    # Latest stable version from PyPI (jq preferred; grep fallback). Best-effort.
    local latest pypi_json
    pypi_json=$(curl -sf --max-time 8 https://pypi.org/pypi/vllm/json 2>/dev/null)
    if [ -n "$pypi_json" ]; then
        if command -v jq >/dev/null 2>&1; then
            latest=$(printf '%s' "$pypi_json" | jq -r '.info.version' 2>/dev/null)
        else
            latest=$(printf '%s' "$pypi_json" \
                | grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 \
                | sed -E 's/.*"([^"]*)"$/\1/')
        fi
    fi
    # Fallback to pip's own index query if PyPI JSON was unreachable.
    [ -z "${latest:-}" ] && latest=$("$pip" index versions vllm 2>/dev/null \
        | sed -n 's/.*LATEST:[[:space:]]*//p' | head -1)

    if [ -z "${latest:-}" ]; then
        echo "ℹ️  vLLM $cur installed — couldn't reach PyPI to check for updates; keeping current."
        return 0
    fi

    # Upgrade only if $latest is strictly newer than $cur (version-aware sort), so
    # a locally-installed nightly is never downgraded to the PyPI stable.
    local newest
    newest=$(printf '%s\n%s\n' "$cur" "$latest" | sort -V 2>/dev/null | tail -1)
    if [ "$cur" = "$latest" ] || [ "$newest" = "$cur" ]; then
        echo "✅ vLLM is up to date ($cur; PyPI latest $latest)."
        return 0
    fi

    echo "⬆️  vLLM $cur → $latest available — upgrading…"

    # Snapshot torch BEFORE pip runs. `pip install -U vllm` resolves torch against
    # PyPI, which does not carry NVIDIA's DGX Spark aarch64+CUDA build — so an
    # upgrade can silently swap in a generic wheel and leave the box GPU-blind.
    # torch.version.cuda is build metadata (independent of whether the driver is
    # currently up), which makes it the right signal for "did we lose CUDA?".
    local pre_torch pre_cuda
    pre_torch=$(_torch_gpu_probe "$py" | awk '$1=="torch"{print $2}')
    pre_cuda=$(_torch_gpu_probe  "$py" | awk '$1=="cuda"{print $2}')
    _stamp_known_good_torch "$VENV_DIR"

    if "$pip" install -U vllm 2>&1 | tail -4; then
        local new
        new=$("$py" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)
        echo "✅ vLLM upgraded to ${new:-$latest}."
    else
        echo "⚠️  vLLM upgrade failed — continuing with $cur."
    fi

    # Did the upgrade strip CUDA support out of torch? If so, put it back.
    local post_torch post_cuda
    post_torch=$(_torch_gpu_probe "$py" | awk '$1=="torch"{print $2}')
    post_cuda=$(_torch_gpu_probe  "$py" | awk '$1=="cuda"{print $2}')
    if [ "$pre_cuda" != "-" ] && [ "$post_cuda" = "-" ]; then
        echo "🚨 The vLLM upgrade replaced torch $pre_torch (CUDA $pre_cuda) with"
        echo "   $post_torch (no CUDA runtime) — that would leave the box GPU-blind."
        echo "🔧 Rolling torch back to $pre_torch…"
        "$pip" install --no-deps --force-reinstall "torch==$pre_torch" 2>&1 | tail -3
        local chk
        chk=$(_torch_gpu_probe "$py" | awk '$1=="cuda"{print $2}')
        if [ "$chk" != "-" ]; then
            echo "✅ torch rolled back — CUDA $chk restored."
            echo "   ⚠️  vLLM $new may now expect a newer torch. If it misbehaves, pin"
            echo "       vLLM back to $cur and leave AUTO_UPDATE_VLLM=false."
        else
            echo "❌ Rollback did not restore CUDA support — see the GPU pre-flight below."
        fi
    fi
    # An upgrade drags flashinfer-python forward but leaves its companion packages
    # behind — reconcile now, while we still know an upgrade just happened.
    _fix_flashinfer_versions "$VENV_DIR"
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU VISIBILITY GUARD
#
# Symptom this exists for:
#   UserWarning: Can't initialize NVML
#   Triton is installed but 0 active driver(s) found (expected 1)
#   No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
#   RuntimeError: Failed to infer device type
# vLLM builds its CLI parser by instantiating a default VllmConfig, and
# DeviceConfig.__post_init__ probes for a device — so with no visible GPU it dies
# during argument parsing, before any model, config, or memory is involved.
#
# Two root causes that look identical in the log but need opposite fixes:
#   1. DRIVER DOWN — the host's nvidia kernel module isn't loaded (or was rebuilt
#      against a different kernel by an apt upgrade). nvidia-smi fails too.
#   2. TORCH BLIND — the driver is fine and nvidia-smi works, but the venv's torch
#      has no CUDA runtime, because an unpinned `pip install -U vllm` resolved
#      torch against PyPI and replaced NVIDIA's aarch64+CUDA build.
# `nvidia-smi` is what separates them, so that is the first thing checked.
# ─────────────────────────────────────────────────────────────────────────────

# True when the driver responds. On GB10 nvidia-smi reports no memory.total, so
# only the exit status is trusted here — never parsed field values.
_gpu_driver_ok() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L >/dev/null 2>&1
}

# Emits "torch <ver|->", "cuda <ver|->", "avail <1|0>", "count <n>" on stdout.
# torch.version.cuda is BUILD metadata and stays populated even when the driver is
# down — that is what makes it a reliable way to tell a clobbered wheel (cuda "-")
# apart from a driver outage (cuda set, avail 0). Stderr is dropped because the
# NVML warnings are exactly what we are diagnosing.
_torch_gpu_probe() {
    local py="$1"
    [ -x "$py" ] || { echo "torch -"; echo "cuda -"; echo "avail 0"; echo "count 0"; return 0; }
    "$py" - <<'PY' 2>/dev/null
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.version.cuda or "-")
    try:
        print("avail", 1 if torch.cuda.is_available() else 0)
        print("count", torch.cuda.device_count())
    except Exception:
        print("avail 0"); print("count 0")
except Exception:
    print("torch -"); print("cuda -"); print("avail 0"); print("count 0")
PY
}

# Cause 1: try to bring the driver back. Loading the modules is safe and idempotent;
# a kernel/driver version mismatch is NOT something to paper over, so it is
# diagnosed and reported rather than guessed at.
_repair_gpu_driver() {
    echo "🔧 GPU driver not responding — attempting recovery…"

    if ! lsmod 2>/dev/null | grep -q '^nvidia'; then
        echo "   nvidia kernel module is not loaded — trying modprobe…"
        # Surface modprobe's error. It names the cause outright ("Invalid module
        # format" = built for another kernel, "Key was rejected by service" =
        # Secure Boot, "Module nvidia not found" = not built for this kernel at
        # all) — swallowing it is what makes this failure feel mysterious.
        local mod mp_err
        for mod in nvidia nvidia_uvm nvidia_modeset; do
            mp_err=$( { sudo -n modprobe "$mod" || modprobe "$mod"; } 2>&1 )
            if [ -n "$mp_err" ]; then
                echo "     modprobe $mod: $mp_err"
            fi
        done
        sleep 2
    else
        echo "   nvidia module IS loaded but NVML still fails — likely a version mismatch."
    fi

    if _gpu_driver_ok; then
        echo "✅ GPU driver recovered — nvidia-smi responds."
        return 0
    fi

    # Still down. Three distinguishable states, in order of how specific the fix is:
    #   a) modinfo finds no nvidia module  → nothing is BUILT for the running kernel
    #   b) running != installed version    → module built against a different kernel
    #   c) both agree but NVML still fails → device/permission level, not versions
    local running installed kver
    kver=$(uname -r 2>/dev/null)
    running=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)
    installed=$(modinfo nvidia 2>/dev/null | sed -n 's/^version: *//p' | head -1)

    echo "❌ GPU driver is still down."
    echo "   running kernel  : ${kver:-unknown}"
    echo "   running driver  : ${running:-<none — module not loaded>}"
    echo "   installed module: ${installed:-<none — modinfo finds no nvidia module>}"

    if [ -z "$installed" ]; then
        # Nothing to load. Either DKMS never built for this kernel, or the driver
        # package is gone. Show which kernels DO have a module — if another kernel
        # has one, the box simply booted the wrong kernel.
        echo "   ⚠️  No nvidia module exists for kernel ${kver:-?}."
        local built
        built=$(ls -d /lib/modules/*/updates/dkms/nvidia.ko* 2>/dev/null \
                | sed 's|/lib/modules/||; s|/updates.*||' | sort -u | tr '\n' ' ')
        if [ -n "$built" ]; then
            echo "       Modules DO exist for: $built"
            echo "       You likely booted a kernel the driver was never built for."
            echo "       Fix:  sudo dkms autoinstall -k $kver && sudo modprobe nvidia"
            echo "       Or boot one of the kernels listed above."
        else
            echo "       No DKMS-built nvidia module for ANY installed kernel."
            echo "       The driver package is missing or its build failed."
            echo "       Check:  dkms status  |  apt list --installed '*nvidia*'"
        fi
        echo "   Then:  sudo dkms autoinstall && sudo reboot"
    elif [ -n "$running" ] && [ "$running" != "$installed" ]; then
        echo "   ⚠️  Mismatch — the module was rebuilt against a different kernel."
        echo "       Fix:  sudo dkms autoinstall && sudo reboot"
    else
        echo "   Module exists but will not bind. Check dmesg for the real reason:"
        echo "       dmesg | grep -i nvidia | tail -30"
        echo "   Common causes: Secure Boot rejecting an unsigned module, or the"
        echo "   device being claimed/blocked (check 'lspci -k' for the driver in use)."
    fi
    echo "   Full detail:  dkms status ; dmesg | grep -i nvidia | tail -30"
    return 1
}

# Cause 2: driver is fine, torch lost its CUDA runtime. Restore the exact version
# recorded before pip last touched the venv — no guessing at wheel URLs.
_repair_torch_cuda() {
    local venv="$1"
    local pip="$venv/bin/pip" py="$venv/bin/python"

    if [ -f "$TORCH_STAMP" ]; then
        local good
        good=$(head -1 "$TORCH_STAMP" 2>/dev/null)
        if [ -n "$good" ]; then
            echo "🔧 Restoring the known-good torch recorded at $TORCH_STAMP: $good"
            # --no-deps so restoring torch cannot cascade into the rest of the stack.
            "$pip" install --no-deps --force-reinstall "torch==$good" 2>&1 | tail -3
            local cuda_now
            cuda_now=$(_torch_gpu_probe "$py" | awk '$1=="cuda"{print $2}')
            if [ "$cuda_now" != "-" ]; then
                echo "✅ torch restored with CUDA support ($cuda_now)."
                return 0
            fi
            echo "⚠️  Restore did not bring CUDA support back."
        fi
    else
        echo "ℹ️  No known-good torch recorded at $TORCH_STAMP — cannot restore by version."
    fi

    if [ "$AUTO_REPAIR_TORCH" = "true" ]; then
        echo "🔧 AUTO_REPAIR_TORCH=true — re-running the DGX Spark vendor installer…"
        curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
        local cuda_now
        cuda_now=$(_torch_gpu_probe "$py" | awk '$1=="cuda"{print $2}')
        [ "$cuda_now" != "-" ] && { echo "✅ Vendor installer restored CUDA torch ($cuda_now)."; return 0; }
    else
        echo "   Set AUTO_REPAIR_TORCH=true to let this script re-run the vendor installer,"
        echo "   or reinstall torch yourself from NVIDIA's index (PyPI does not carry the"
        echo "   DGX Spark aarch64+CUDA build):"
        echo "     curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash"
    fi
    return 1
}

# Entry point. Returns non-zero when the GPU cannot be made visible.
# Usage: _preflight_gpu <venv_dir>
_preflight_gpu() {
    local venv="$1"
    local py="$venv/bin/python"

    local probe torch_ver cuda_ver avail count
    probe=$(_torch_gpu_probe "$py")
    torch_ver=$(echo "$probe" | awk '$1=="torch"{print $2}')
    cuda_ver=$(echo  "$probe" | awk '$1=="cuda"{print $2}')
    avail=$(echo     "$probe" | awk '$1=="avail"{print $2}')
    count=$(echo     "$probe" | awk '$1=="count"{print $2}')

    # Fast path: everything works.
    if [ "$avail" = "1" ] && [ "${count:-0}" -gt 0 ]; then
        echo "✅ GPU visible: ${count} device(s), torch $torch_ver (CUDA $cuda_ver)"
        return 0
    fi

    echo "⚠️  No GPU visible to torch — vLLM would fail with 'Failed to infer device type'."
    echo "        torch          $torch_ver"
    echo "        torch CUDA     $cuda_ver"
    echo "        devices seen   ${count:-0}"

    if [ "$AUTO_REPAIR_GPU" != "true" ]; then
        echo "   AUTO_REPAIR_GPU=false — not attempting recovery."
        return 1
    fi

    # nvidia-smi decides which of the two causes this is.
    if ! _gpu_driver_ok; then
        _repair_gpu_driver || return 1
    else
        echo "✅ nvidia-smi responds — the driver is fine, so this is the venv's torch."
        if [ "$cuda_ver" = "-" ]; then
            echo "   torch has NO CUDA runtime — its wheel was replaced with a generic build."
            _repair_torch_cuda "$venv" || return 1
        else
            echo "   torch is a CUDA build but still sees no device."
            echo "   Check container/cgroup GPU access, CUDA_VISIBLE_DEVICES, and permissions"
            echo "   on /dev/nvidia*. Current CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"
            return 1
        fi
    fi

    # Re-probe after whatever repair ran.
    probe=$(_torch_gpu_probe "$py")
    avail=$(echo "$probe" | awk '$1=="avail"{print $2}')
    count=$(echo "$probe" | awk '$1=="count"{print $2}')
    if [ "$avail" = "1" ] && [ "${count:-0}" -gt 0 ]; then
        echo "✅ GPU recovered: ${count} device(s) now visible."
        return 0
    fi

    echo "❌ GPU still not visible after repair — not starting models."
    return 1
}

# Write a pip constraints file pinning the CURRENTLY INSTALLED torch stack, and
# echo its path. Using `pip install -c <file> …` instead of `--no-deps` is the
# difference between "protect torch" and "protect torch AND resolve everything
# else correctly".
#
# Why this exists: --no-deps stopped pip from replacing NVIDIA's DGX Spark torch,
# but it ALSO stopped pip from adjusting flashinfer's sibling packages. Pinning
# flashinfer-python down to 0.6.13 that way left nvidia-cutlass-dsl at the version
# 0.6.14 wanted, and the engine then died at load time with
#   AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
# — flashinfer calling a CuTe DSL API its sibling no longer exposed. A constraints
# file fixes only torch and lets the resolver do its job on the rest.
# Usage: cfile=$(_torch_constraints_file "$venv")
_torch_constraints_file() {
    local venv="$1"
    local py="$venv/bin/python"
    local cfile="${TMPDIR:-/tmp}/vllm-torch-constraints.txt"
    : > "$cfile"
    # Bare "torch==X.Y.Z" still matches a local build like 2.11.0+cu130 (PEP 440
    # ignores the local segment when the specifier omits it), so the NVIDIA wheel
    # stays put instead of being "upgraded" to a PyPI generic of the same version.
    "$py" - >> "$cfile" 2>/dev/null <<'PY'
from importlib.metadata import version, PackageNotFoundError
for pkg in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{pkg}=={version(pkg).split('+')[0]}")
    except PackageNotFoundError:
        pass
PY
    printf '%s' "$cfile"
}

# Record the current torch as known-good, so a later clobber can be undone.
# Only stamps a torch that actually has a CUDA runtime.
_stamp_known_good_torch() {
    local venv="$1"
    local py="$venv/bin/python"
    local probe tv cv
    probe=$(_torch_gpu_probe "$py")
    tv=$(echo "$probe" | awk '$1=="torch"{print $2}')
    cv=$(echo "$probe" | awk '$1=="cuda"{print $2}')
    if [ -n "$tv" ] && [ "$tv" != "-" ] && [ "$cv" != "-" ]; then
        mkdir -p "$(dirname "$TORCH_STAMP")" 2>/dev/null || true
        printf '%s\n' "$tv" > "$TORCH_STAMP" 2>/dev/null || true
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# vLLM PIN ALIGNMENT
#
# vLLM declares exact "==" pins for the accelerator packages it was built and
# tested against. When any of them drifts, the failure surfaces deep inside engine
# startup as a missing symbol rather than as a version complaint — e.g.
#   AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
# which was nvidia-cutlass-dsl sitting at 4.6.1 (with its cu13 libs stranded at
# 4.6.0) while vllm 0.26.0 pins nvidia-cutlass-dsl[cu13]==4.6.0.
#
# Special-casing flashinfer was too narrow: the same class of drift hit
# nvidia-cutlass-dsl and apache-tvm-ffi in the same venv. This reconciles EVERY
# unsatisfied "==" pin vLLM declares, in one pass, under the torch constraints file
# so the DGX Spark build is never collateral damage.
#
# Extras are preserved (nvidia-cutlass-dsl[cu13]==4.6.0, not the bare package) —
# installing without the extra is what leaves a split base/cu13 install behind.
# Usage: _align_vllm_pins <venv_dir>
# ─────────────────────────────────────────────────────────────────────────────
_align_vllm_pins() {
    local venv="$1"
    local py="$venv/bin/python" pip="$venv/bin/pip"
    [ -x "$py" ] || return 0

    # When _repair_cutlass_stack proved vLLM's cutlass pin is metadata-stale (a
    # clean install at the pin lacked a symbol the runtime needs) it records the
    # working version in this marker. Realigning to the pin would reintroduce
    # the exact crash the override fixed — so cutlass is skipped while it exists.
    local cutlass_override=""
    [ -f "$venv/.cutlass-dsl-override" ] && cutlass_override=$(head -1 "$venv/.cutlass-dsl-override")
    export _VLLM_PIN_SKIP_CUTLASS="${cutlass_override:+1}"

    # Emit one "<name><extras>==<version>" per UNSATISFIED exact pin.
    # flashinfer packages are excluded — _fix_flashinfer_versions owns those, and
    # vLLM's flashinfer pin is knowingly unsatisfiable (no cubin wheel published).
    local unmet
    unmet=$("$py" - <<'PY' 2>/dev/null
import re
from importlib.metadata import requires, version, PackageNotFoundError
try:
    from packaging.requirements import Requirement
except Exception:
    raise SystemExit(0)
try:
    reqs = requires("vllm") or []
except PackageNotFoundError:
    raise SystemExit(0)
for raw in reqs:
    try:
        r = Requirement(raw)
    except Exception:
        continue
    # Skip requirements gated behind a marker that does not apply here.
    if r.marker is not None and not r.marker.evaluate():
        continue
    if "flashinfer" in r.name.lower():
        continue
    # cutlass override active: _repair_cutlass_stack proved the pin stale.
    import os
    if os.environ.get("_VLLM_PIN_SKIP_CUTLASS") == "1" and r.name.lower() == "nvidia-cutlass-dsl":
        continue
    # torch family is owned by the constraints file — and its installed versions
    # carry a local suffix (2.11.0+cu130) that a naive == comparison reads as
    # drift, which would try to "restore" NVIDIA's build to a PyPI generic.
    if r.name.lower() in ("torch", "torchvision", "torchaudio"):
        continue
    pins = [s for s in r.specifier if s.operator == "=="]
    if len(pins) != 1:
        continue
    want = pins[0].version
    try:
        have = version(r.name)
    except PackageNotFoundError:
        continue          # not installed at all — not our business to add it
    # Compare public versions so 2.11.0+cu130 satisfies ==2.11.0 (PEP 440).
    try:
        from packaging.version import Version
        if Version(have).public == Version(want).public:
            continue
    except Exception:
        if have == want:
            continue
    extras = f"[{','.join(sorted(r.extras))}]" if r.extras else ""
    print(f"{r.name}{extras}=={want}")
PY
)
    [ -z "$unmet" ] && { echo "✅ vLLM's pinned dependencies are all satisfied"; return 0; }

    echo "⚠️  Packages drifted off vLLM's pinned versions — these cause missing-symbol"
    echo "    crashes deep in engine startup, not clean version errors:"
    printf '%s\n' "$unmet" | sed 's/^/        /'

    local -a specs=()
    local line
    while IFS= read -r line; do
        [ -n "$line" ] && specs+=("$line")
    done <<< "$unmet"
    [ "${#specs[@]}" -eq 0 ] && return 0

    local cfile
    cfile=$(_torch_constraints_file "$venv")
    echo "🔧 Restoring vLLM's pins (torch held by constraints) ..."
    if "$pip" install -c "$cfile" "${specs[@]}" 2>&1 | tail -4 | sed 's/^/   | /'; then
        echo "✅ vLLM pin alignment done."
    else
        echo "⚠️  Some pins could not be restored — see pip output above."
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# FLASHINFER VERSION GUARD
# flashinfer ships as three separately-versioned PyPI packages — flashinfer-python,
# flashinfer-cubin, flashinfer-jit-cache — and hard-refuses to import unless their
# versions match exactly. `pip install -U vllm` bumps flashinfer-python (a declared
# dependency) but leaves the other two at their old version, so AUTO_UPDATE_VLLM
# can silently poison the *next* serve with:
#   RuntimeError: flashinfer-cubin version (0.6.13) does not match
#                 flashinfer version (0.6.14).
# vLLM imports flashinfer unconditionally from Sampler.__init__, so this kills
# EngineCore during init_device() — long before KV-cache profiling. It reads like
# an OOM or a bad model (and sends you chasing --gpu-memory-utilization), but no
# GPU memory was ever allocated.
#
# This aligns the companions to flashinfer-python; if no matching companion wheel
# exists, it pins flashinfer-python back down to what is already on disk instead.
# Installs go through a pip CONSTRAINTS file that pins the installed torch, so the
# hardware-specific DGX Spark build cannot be replaced (same concern as
# AUTO_UPDATE_VLLM above) while flashinfer's own siblings — nvidia-cutlass-dsl in
# particular — are still allowed to move to the versions it needs.
# Cheap no-op when versions already agree, so it is safe to call on every run.
# Usage: _fix_flashinfer_versions <venv_dir>
# ─────────────────────────────────────────────────────────────────────────────
_fix_flashinfer_versions() {
    local venv="$1"
    local py="$venv/bin/python" pip="$venv/bin/pip"
    if [ ! -x "$py" ]; then
        echo "⚠️  flashinfer check skipped — no python at $py"
        return 0
    fi

    # One "<package> <version|->" line each; "-" means not installed.
    local report
    report=$("$py" - <<'PY' 2>/dev/null
from importlib.metadata import version, PackageNotFoundError
for pkg in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
    try:
        print(pkg, version(pkg))
    except PackageNotFoundError:
        print(pkg, "-")
PY
)
    if [ -z "$report" ]; then
        echo "⚠️  Could not query flashinfer versions in $venv — skipping reconciliation."
        return 0
    fi

    local core cubin jitcache
    core=$(echo     "$report" | awk '$1=="flashinfer-python"{print $2}')
    cubin=$(echo    "$report" | awk '$1=="flashinfer-cubin"{print $2}')
    jitcache=$(echo "$report" | awk '$1=="flashinfer-jit-cache"{print $2}')

    if [ -z "$core" ] || [ "$core" = "-" ]; then
        echo "ℹ️  flashinfer-python not installed in $venv — nothing to reconcile."
        return 0
    fi

    # Prefer the version vLLM actually pins over whatever flashinfer-python happens
    # to be. Aligning the trio to each other is enough to make `import flashinfer`
    # work, but it can settle on a version vLLM does not want — which is how the
    # venv ended up self-consistent at 0.6.13 while vllm 0.26.0 required 0.6.14,
    # leaving `pip check` unhappy and the stack subtly off-spec.
    local vllm_pin
    vllm_pin=$("$py" - <<'PY' 2>/dev/null
import re
from importlib.metadata import requires, PackageNotFoundError
try:
    reqs = requires("vllm") or []
except PackageNotFoundError:
    reqs = []
for r in reqs:
    base = r.split(";")[0].strip()          # drop environment markers/extras
    m = re.match(r'^flashinfer[-_]python\s*==\s*([0-9][^\s,]*)$', base)
    if m:
        print(m.group(1))
        break
PY
)

    local target="$core" target_src="installed flashinfer-python"
    local target_unsatisfiable=0
    # Declared here (not inside the reconcile block) so the advice messages at the
    # end can reference it even when no reconcile was needed. set -u is on.
    local cfile
    cfile=$(_torch_constraints_file "$venv")
    if [ -n "$vllm_pin" ]; then
        target="$vllm_pin"
        target_src="vLLM's pinned requirement"
    fi

    # Every installed member of the trio that disagrees with the target.
    # Built as a real array — never a space-joined string relying on word-splitting.
    local -a pins=()
    [ "$core" != "$target" ] && pins+=("flashinfer-python==$target")
    if [ -n "$cubin" ] && [ "$cubin" != "-" ] && [ "$cubin" != "$target" ]; then
        pins+=("flashinfer-cubin==$target")
    fi
    if [ -n "$jitcache" ] && [ "$jitcache" != "-" ] && [ "$jitcache" != "$target" ]; then
        pins+=("flashinfer-jit-cache==$target")
    fi

    if [ "${#pins[@]}" -gt 0 ]; then
        echo "⚠️  flashinfer versions need reconciling (target $target — $target_src):"
        printf '        flashinfer-python     %s\n' "$core"
        printf '        flashinfer-cubin      %s\n' "$cubin"
        printf '        flashinfer-jit-cache  %s\n' "$jitcache"
        echo "🔧 Aligning the trio to $target ..."

        # Constraints (not --no-deps): torch is pinned, but flashinfer's siblings —
        # notably nvidia-cutlass-dsl — are allowed to move to the versions this
        # flashinfer actually needs. See _torch_constraints_file for the failure
        # that --no-deps caused here.
        local align_out align_rc
        align_out=$("$pip" install -c "$cfile" "${pins[@]}" 2>&1); align_rc=$?
        echo "$align_out" | tail -4 | sed 's/^/   | /'
        # "No matching distribution found" means the version was never PUBLISHED —
        # a different situation from a transient pip failure, and one no command the
        # user runs can fix. Tracked so the advice below stays honest.
        if echo "$align_out" | grep -q "No matching distribution found"; then
            target_unsatisfiable=1
        fi
        if [ "$align_rc" -ne 0 ]; then
            # If the ONLY thing blocking vLLM's pin is a companion with no published
            # wheel, removing that companion is right and downgrading is wrong.
            # flashinfer-cubin is an optional prebuilt-kernel cache, not a required
            # sibling — vllm 0.26.0 pins flashinfer-python==0.6.14 and no cubin
            # 0.6.14 was ever published, because that install simply has no cubin.
            # Downgrading flashinfer-python to match a stale cubin instead is what
            # produced "module 'cutlass.cute.core' has no attribute 'ThrMma'":
            # 0.6.13 calls a CuTe API that nvidia-cutlass-dsl 4.6.0 (vLLM's pin)
            # does not expose. Prefer dropping cubin, then re-try the pin.
            if [ "$target_unsatisfiable" -eq 1 ] && [ "$target" = "$vllm_pin" ] \
               && [ -n "$cubin" ] && [ "$cubin" != "-" ]; then
                echo "⚠️  No flashinfer-cubin wheel at $target — but cubin is an optional"
                echo "    kernel cache, not a required sibling. Removing it and retrying"
                echo "    at vLLM's pin instead of downgrading flashinfer-python."
                if "$pip" uninstall -y flashinfer-cubin >/dev/null 2>&1 \
                   && "$pip" install -c "$cfile" "flashinfer-python==$target" >/dev/null 2>&1; then
                    echo "✅ flashinfer-python pinned to $target with cubin removed."
                    echo "   First launch will JIT-compile kernels (slower once, then cached)."
                    target_unsatisfiable=0
                    align_rc=0
                fi
            fi
        fi
        if [ "$align_rc" -ne 0 ]; then
            # Ladder down: vLLM's pin has no wheel here → try the installed
            # flashinfer-python version → finally pin down to a companion's version.
            local fallback=""
            if [ "$target" != "$core" ]; then
                echo "⚠️  No wheels at $target — retrying at the installed flashinfer-python $core."
                fallback="$core"
            else
                local floor="$cubin"
                if [ -z "$floor" ] || [ "$floor" = "-" ]; then
                    floor="$jitcache"
                fi
                [ -n "$floor" ] && [ "$floor" != "-" ] && fallback="$floor"
                [ -n "$fallback" ] && echo "⚠️  No wheels at $target — pinning down to $fallback instead."
            fi
            if [ -n "$fallback" ]; then
                local -a fb=()
                [ "$core"     != "$fallback" ] && fb+=("flashinfer-python==$fallback")
                [ -n "$cubin" ]    && [ "$cubin"    != "-" ] && [ "$cubin"    != "$fallback" ] && fb+=("flashinfer-cubin==$fallback")
                [ -n "$jitcache" ] && [ "$jitcache" != "-" ] && [ "$jitcache" != "$fallback" ] && fb+=("flashinfer-jit-cache==$fallback")
                [ "${#fb[@]}" -gt 0 ] && { "$pip" install -c "$cfile" "${fb[@]}" || true; }
            fi
        fi
    fi

    # Authoritative check: the import is what actually enforces the version match.
    if "$py" -c 'import flashinfer' >/dev/null 2>&1; then
        local final
        final=$("$py" -c 'from importlib.metadata import version; print(version("flashinfer-python"))' 2>/dev/null)
        if [ -n "$vllm_pin" ] && [ -n "$final" ] && [ "$final" != "$vllm_pin" ]; then
            # Self-consistent, so it imports and will serve — but off vLLM's spec.
            # Warn rather than fail: this state works, and failing here would block
            # a box that is otherwise fine.
            echo "✅ flashinfer imports cleanly (version $final)"
            if [ "$target_unsatisfiable" -eq 1 ]; then
                # vLLM pins a companion version that was never published. Nothing the
                # user can run fixes this, so do NOT suggest a command that must fail.
                echo "   ℹ️  vLLM pins flashinfer-python==$vllm_pin, but no matching"
                echo "       flashinfer-cubin/jit-cache wheel exists on PyPI at that version."
                echo "       $final is the newest fully-published set, so the trio stays there."
                echo "       Nothing to do — 'pip check' will keep flagging this until upstream"
                echo "       publishes matching wheels or vLLM relaxes the pin. It runs fine."
            else
                echo "   ⚠️  vLLM pins flashinfer-python==$vllm_pin but the trio settled at $final."
                echo "       Self-consistent so it runs, but 'pip check' will flag it and the"
                echo "       combination is untested upstream. Resolve with:"
                echo "         $pip install -c $cfile flashinfer-python==$vllm_pin \\"
                echo "             flashinfer-cubin==$vllm_pin flashinfer-jit-cache==$vllm_pin"
            fi
            return 0
        fi
        echo "✅ flashinfer imports cleanly (version ${final:-unknown})"
        return 0
    fi

    echo "❌ flashinfer still fails to import — vLLM will not start. Error:"
    "$py" -c 'import flashinfer' 2>&1 | tail -5 | sed 's/^/   | /'
    echo "   Fix manually, e.g.:"
    echo "     $pip install -c $cfile flashinfer-python==$target \\"
    echo "         flashinfer-cubin==$target flashinfer-jit-cache==$target"
    echo "   Last resort (unsafe — mismatched cubins can fail mid-request rather than"
    echo "   at startup): export FLASHINFER_DISABLE_VERSION_CHECK=1"
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# CUTLASS-DSL STACK REPAIR
#
# Symptom: EngineCore dies at model load with
#   AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
# (or another missing cutlass symbol) EVEN AFTER every nvidia-cutlass-dsl dist
# shows the correct pinned version. Cause: the `cutlass` package directory has
# been written by several versions in sequence (install 4.6.0 → force-upgrade
# 4.6.1 → downgrade 4.6.0). In-place up/downgrades only remove files listed in
# the outgoing dist's RECORD — files that moved or were renamed between versions
# survive as orphans, and Python imports whatever is on disk. The result is
# version-correct metadata sitting on a mixed package directory.
#
# Repair = the only reliable one: uninstall every cutlass-dsl dist, delete any
# leftover cutlass* directories from site-packages, then install vLLM's exact
# pin (with its extras — [cu13] — since installing the bare name is what leaves
# the CUDA-variant libs behind) under the torch constraints file.
# Usage: _repair_cutlass_stack <venv_dir>
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# MISSING-SYMBOL CONSUMER REPAIR
#
# _repair_cutlass_stack assumes cutlass-dsl itself is behind. That assumption
# broke in the wild: "module 'cutlass.cute.core' has no attribute 'ThrMma'"
# persisted after a clean install at BOTH vLLM's pin (4.6.0) and the newest
# release on PyPI (4.6.1) — so no published cutlass-dsl has ThrMma. The engine
# traceback (see _show_log_tail) named the actual culprit: NVIDIA's `quack`
# kernel library references cute.core.ThrMma in a type annotation, evaluated at
# import time. quack — not cutlass-dsl — is the package that's stale/mismatched.
#
# This is the general form of that lesson: when the symbol-defining package
# (cutlass-dsl) checks out clean at every published version, the bug is more
# likely in whichever CONSUMER package's code references that symbol. Rather
# than hardcode "quack", this parses the engine traceback for the last
# site-packages/<pkg>/ frame before the crash, maps it to its distribution name,
# and reinstalls that fresh — letting pip's own resolver reconcile it against
# whatever cutlass-dsl is currently installed.
# Usage: _repair_symbol_consumer <venv_dir> <log_file>
# ─────────────────────────────────────────────────────────────────────────────
_repair_symbol_consumer() {
    local venv="$1" log_file="$2"
    local py="$venv/bin/python" pip="$venv/bin/pip"
    [ -x "$pip" ] && [ -f "$log_file" ] || return 1

    # Last non-infrastructure package mentioned in a traceback "File" frame is
    # the most specific consumer of the missing symbol — infra packages
    # (vllm/torch/cutlass/flashinfer/etc.) are excluded since repairing those is
    # what the other tiers already do.
    local pkg
    pkg=$(grep -aoE 'site-packages/[A-Za-z0-9_]+/' "$log_file" \
          | sed -E 's#site-packages/([A-Za-z0-9_]+)/#\1#' \
          | grep -avE '^(vllm|torch|torchvision|torchaudio|cutlass|flashinfer|triton|uvloop|asyncio|apache_tvm_ffi)$' \
          | tail -1)
    [ -z "$pkg" ] && { echo "   ℹ️  No non-infrastructure package found in the traceback."; return 1; }

    local dist
    dist=$("$py" - "$pkg" <<'PY' 2>/dev/null
import sys
from importlib.metadata import packages_distributions
mod = sys.argv[1]
dists = packages_distributions().get(mod)
print(dists[0] if dists else mod)
PY
)
    [ -z "$dist" ] && dist="$pkg"

    echo "   🔎 Traceback shows '$pkg' (package: $dist) as the code that references the"
    echo "       missing symbol. cutlass-dsl checked out clean at every published"
    echo "       version, so trying a fresh reinstall of $dist instead."

    local cfile
    cfile=$(_torch_constraints_file "$venv")
    local cache_dir
    for cache_dir in "$HOME/.cache/flashinfer" "$HOME/.cache/vllm" /root/.cache/flashinfer /root/.cache/vllm; do
        [ -d "$cache_dir" ] && rm -rf "$cache_dir"
    done
    rm -rf /tmp/torchinductor_* 2>/dev/null

    "$pip" uninstall -y "$dist" >/dev/null 2>&1
    if ! "$pip" install -c "$cfile" -U "$dist" 2>&1 | tail -4 | sed 's/^/   | /'; then
        echo "❌ Could not reinstall $dist."
        return 1
    fi

    # Re-verify against the EXACT symbol the original crash needed, not just a
    # bare import — the module importing cleanly proved nothing last time either.
    local miss_line miss_mod miss_attr
    miss_line=$(grep -aoE "module '[A-Za-z0-9_.]+' has no attribute '[A-Za-z0-9_]+'" "$log_file" | tail -1)
    miss_mod=$(echo  "$miss_line" | sed -n "s/module '\([^']*\)' has no attribute.*/\1/p")
    miss_attr=$(echo "$miss_line" | sed -n "s/.*has no attribute '\([^']*\)'.*/\1/p")

    if [ -n "$miss_mod" ] && [ -n "$miss_attr" ]; then
        if "$py" -c "import importlib; m=importlib.import_module('$miss_mod'); getattr(m,'$miss_attr')" >/dev/null 2>&1; then
            echo "✅ $dist reinstall resolved ${miss_mod}.${miss_attr}."
            return 0
        fi
        echo "❌ ${miss_mod}.${miss_attr} still missing after reinstalling $dist."
        return 1
    fi
    if "$py" -c "import $pkg" >/dev/null 2>&1; then
        echo "✅ $dist reinstalled and imports cleanly."
        return 0
    fi
    echo "❌ $dist still fails to import after reinstall."
    return 1
}

_repair_cutlass_stack() {
    local venv="$1" log_file="${2:-}"
    local py="$venv/bin/python" pip="$venv/bin/pip"
    [ -x "$pip" ] || { echo "⚠️  cutlass repair skipped — no pip at $pip"; return 1; }

    # Pull the exact missing symbol out of the crash ("module 'X' has no
    # attribute 'Y'") so success can be verified against the REAL requirement.
    # `import cutlass.cute.core` succeeding proves nothing — the module imports
    # fine in every version; it's the attribute that's version-dependent.
    local miss_mod="" miss_attr=""
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        local miss_line
        miss_line=$(grep -aoE "module '[A-Za-z0-9_.]+' has no attribute '[A-Za-z0-9_]+'" "$log_file" | tail -1)
        miss_mod=$(echo  "$miss_line" | sed -n "s/module '\([^']*\)' has no attribute.*/\1/p")
        miss_attr=$(echo "$miss_line" | sed -n "s/.*has no attribute '\([^']*\)'.*/\1/p")
        [ -n "$miss_attr" ] && echo "   crash requires: ${miss_mod}.${miss_attr}"
    fi

    # Verify a module (and optionally the attribute the crash wanted) resolves.
    _cutlass_ok() {
        if [ -n "$miss_mod" ] && [ -n "$miss_attr" ]; then
            "$py" -c "import importlib; m=importlib.import_module('$miss_mod'); getattr(m,'$miss_attr')" >/dev/null 2>&1
        else
            "$py" -c 'import cutlass.cute.core' >/dev/null 2>&1
        fi
    }

    # vLLM's exact spec for nvidia-cutlass-dsl, extras included.
    local spec
    spec=$("$py" - <<'PY' 2>/dev/null
from importlib.metadata import requires, PackageNotFoundError
try:
    from packaging.requirements import Requirement
    for raw in (requires("vllm") or []):
        r = Requirement(raw)
        if r.marker is not None and not r.marker.evaluate():
            continue
        if r.name.lower() == "nvidia-cutlass-dsl":
            extras = f"[{','.join(sorted(r.extras))}]" if r.extras else ""
            pins = [s.version for s in r.specifier if s.operator == "=="]
            print(f"{r.name}{extras}=={pins[0]}" if pins else f"{r.name}{extras}")
            break
except Exception:
    pass
PY
)
    [ -z "$spec" ] && spec="nvidia-cutlass-dsl[cu13]"

    echo "🔧 Rebuilding the cutlass-dsl stack from scratch (target: $spec) ..."
    "$pip" uninstall -y \
        nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core \
        nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 2>/dev/null | tail -2

    # Delete orphans the uninstalls leave behind — this is the actual fix; the
    # reinstall alone would rewrite only the files the new version owns.
    local sp
    sp=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)
    if [ -n "$sp" ] && [ -d "$sp" ]; then
        rm -rf "$sp"/cutlass "$sp"/cutlass-* "$sp"/cutlass_* 2>/dev/null
        echo "   cleared leftover cutlass* directories under $sp"
    fi

    # Purge JIT/compile caches. Generated code cached by flashinfer / vLLM /
    # inductor was produced against whatever cutlass API was installed at the
    # time — it survives every pip operation and keeps referencing the old API.
    local cache_dir
    for cache_dir in "$HOME/.cache/flashinfer" "$HOME/.cache/vllm" /root/.cache/flashinfer /root/.cache/vllm; do
        if [ -d "$cache_dir" ]; then
            rm -rf "$cache_dir"
            echo "   purged stale JIT cache: $cache_dir"
        fi
    done
    rm -rf /tmp/torchinductor_* 2>/dev/null

    local cfile
    cfile=$(_torch_constraints_file "$venv")
    if ! "$pip" install -c "$cfile" "$spec" 2>&1 | tail -3 | sed 's/^/   | /'; then
        echo "❌ cutlass-dsl reinstall failed — see pip output above."
        return 1
    fi

    if _cutlass_ok; then
        echo "✅ cutlass-dsl stack rebuilt cleanly ($spec)."
        rm -f "$venv/.cutlass-dsl-override" 2>/dev/null
        return 0
    fi

    # A CLEAN install at vLLM's pin still lacks the symbol the runtime asks for.
    # That means the pin is metadata-stale: the code path that executes (e.g.
    # flashinfer's cute-dsl kernels, whose own requirement is just >=4.5.0) was
    # written against a NEWER cutlass API. Runtime compatibility beats metadata —
    # try the latest cutlass-dsl and keep it if the symbol appears.
    if [ -n "$miss_attr" ]; then
        echo "⚠️  A clean $spec install still lacks ${miss_mod}.${miss_attr}."
        echo "    The pinned version predates that API — trying the latest cutlass-dsl…"
        local bare="${spec%%==*}"     # keep name+extras, drop the stale pin
        "$pip" install -c "$cfile" -U "$bare" 2>&1 | tail -3 | sed 's/^/   | /'
        if _cutlass_ok; then
            local got
            got=$("$py" -c 'from importlib.metadata import version; print(version("nvidia-cutlass-dsl"))' 2>/dev/null)
            echo "✅ cutlass-dsl $got provides ${miss_mod}.${miss_attr} — keeping it."
            echo "   (vLLM's ==${spec##*==} pin is metadata-stale; 'pip check' will complain,"
            echo "    but this is the version the executing code actually needs.)"
            # Marker stops _align_vllm_pins from tug-of-warring it back down.
            printf '%s\n' "${got:-unknown}" > "$venv/.cutlass-dsl-override" 2>/dev/null
            return 0
        fi
    fi

    echo "❌ cutlass symbol still unresolved after clean reinstall + latest version:"
    "$py" -c "import ${miss_mod:-cutlass.cute.core}" 2>&1 | tail -3 | sed 's/^/   | /'
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# VENV REBUILD (final escalation)
# Moves the current vllm-install dir aside, re-runs the DGX Spark vendor
# installer in its parent directory (the installer builds ./vllm-install
# relative to CWD), verifies the fresh venv imports vLLM and sees the GPU, and
# restores the old env if anything fails. Gated on AUTO_REBUILD_VENV, an
# environment-shaped failure signature, and a 24h loop-guard stamp — pass
# "force" as $2 to bypass the gates (FORCE_VLLM_REINSTALL path).
# Usage: _rebuild_vllm_venv [log_file] [force]
# ─────────────────────────────────────────────────────────────────────────────
_rebuild_vllm_venv() {
    local log_file="${1:-}" force="${2:-}"
    local venv="${VENV_DIR:-}"
    [ -n "$venv" ] || return 1

    if [ "$force" != "force" ]; then
        if [ "$AUTO_REBUILD_VENV" != "true" ]; then
            echo "   AUTO_REBUILD_VENV=false — skipping venv rebuild escalation."
            return 1
        fi
        # Only for environment-shaped failures. An OOM or a bad model file is not
        # the venv's fault, and a 15-minute rebuild would fix nothing.
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            grep -aqiE "out of memory|OutOfMemory" "$log_file" && return 1
            grep -aqE "has no attribute|ImportError|ModuleNotFoundError|undefined symbol|version .*does not match|Failed to infer device type" "$log_file" \
                || return 1
        fi
        # Loop guard: at most one automatic rebuild per 24h.
        if [ -f "$REBUILD_STAMP" ]; then
            local age
            age=$(( $(date +%s) - $(stat -c %Y "$REBUILD_STAMP" 2>/dev/null || stat -f %m "$REBUILD_STAMP" 2>/dev/null || echo 0) ))
            if [ "$age" -lt 86400 ]; then
                echo "   ⛔ venv was auto-rebuilt $((age/60)) min ago and models still fail."
                echo "      NOT rebuilding again (loop guard) — the venv is no longer the suspect."
                echo "      Force another rebuild: FORCE_VLLM_REINSTALL=true $0 <args>"
                return 1
            fi
        fi
    fi

    local install_dir parent ts
    install_dir=$(dirname "$venv")      # …/vllm-install
    parent=$(dirname "$install_dir")
    ts=$(date +%Y%m%d-%H%M%S)

    echo ""
    echo "🏗️  ESCALATION: rebuilding the vLLM venv from the vendor installer."
    echo "    Targeted repairs restored vLLM's pinned versions and the failure persists,"
    echo "    so the venv itself is no longer trustworthy. A fresh vendor build is the"
    echo "    tested GB10 state. Takes ~10-15 min (Triton compiles from source)."
    echo "    Old env preserved at: $install_dir.broken-$ts"

    date > "$REBUILD_STAMP" 2>/dev/null || true
    mv "$install_dir" "$install_dir.broken-$ts" 2>/dev/null || true

    if ( cd "$parent" && curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash ); then
        if [ -x "$venv/bin/python" ] && \
           "$venv/bin/python" -c 'import sys, vllm, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
            echo "✅ Fresh venv verified: vLLM imports, GPU visible."
            _stamp_known_good_torch "$venv"
            rm -f "$venv/.cutlass-dsl-override" 2>/dev/null
            return 0
        fi
        echo "❌ Rebuilt venv fails verification (vllm import / GPU visibility)."
    else
        echo "❌ Vendor installer failed."
    fi
    # Roll back so the box is no worse off than before the attempt.
    if [ -d "$install_dir.broken-$ts" ] && [ ! -d "$install_dir" ]; then
        mv "$install_dir.broken-$ts" "$install_dir"
        echo "   Previous venv restored."
    fi
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# POST-MORTEM PLAYBOOK
# Reads a dead model's log, matches it against failure signatures this script
# knows how to fix, and runs the matching repair. Returns 0 only when a repair
# ran AND reported success — the caller then retries the launch once.
# This is the general mechanism the one-off fixes kept approximating by hand:
# the crash tells us which subsystem broke; fix that subsystem, try again.
# Usage: _diagnose_and_repair <log_file>
# ─────────────────────────────────────────────────────────────────────────────
_diagnose_and_repair() {
    local log_file="$1"
    [ -f "$log_file" ] || return 1
    [ -n "${VENV_DIR:-}" ] || return 1

    if grep -aqE "module 'cutlass[^']*' has no attribute|(ImportError|ModuleNotFoundError).*cutlass" "$log_file"; then
        echo "   🔎 Known failure: cutlass-dsl API mismatch — repairing…"
        _repair_cutlass_stack "$VENV_DIR" "$log_file" && return 0
        # cutlass-dsl itself checked out clean at every published version — the
        # symbol truly doesn't exist there. The bug is more likely in whichever
        # package's CODE references it (proven in the wild: NVIDIA's `quack`
        # kernel lib, not cutlass-dsl, was the actual mismatch — see changelog).
        echo "   🔎 cutlass-dsl is not the mismatch — trying the package that calls the symbol…"
        _repair_symbol_consumer "$VENV_DIR" "$log_file" && return 0
        return 1
    fi
    if grep -aq "flashinfer.*version.*does not match" "$log_file"; then
        echo "   🔎 Known failure: flashinfer version skew — repairing…"
        _fix_flashinfer_versions "$VENV_DIR" && return 0
        return 1
    fi
    if grep -aq "Failed to infer device type" "$log_file"; then
        echo "   🔎 Known failure: GPU not visible — running GPU recovery…"
        _preflight_gpu "$VENV_DIR" && return 0
        return 1
    fi
    # Hybrid Mamba/attention models (the Nemotron-3-Nano family) run Mamba cache
    # in "align" mode whenever --enable-prefix-caching is on. That mode computes
    # a block_size from the model's mamba state layout and REQUIRES
    # max_num_batched_tokens >= block_size — vLLM's own default (2048) is often
    # too small, and the required block_size varies by model/context, so no
    # single catalog constant covers every case. This is a launch-ARGS problem,
    # not a venv problem: the fix is retrying with a bigger
    # --max-num-batched-tokens, not reinstalling anything. Sets
    # _VLLM_ARG_OVERRIDE for the caller to append (argparse keeps the LAST value
    # of a repeated flag, so this wins over whatever the catalog block set).
    local mamba_line
    mamba_line=$(grep -aoE 'block_size \([0-9]+\) must be <= max_num_batched_tokens \([0-9]+\)' "$log_file" | tail -1)
    if [ -n "$mamba_line" ]; then
        local need safe
        need=$(echo "$mamba_line" | sed -n 's/.*block_size (\([0-9]*\)).*/\1/p')
        if [ -n "$need" ]; then
            # Round up to the next multiple of 1024 above the requirement, so
            # small model-len/context changes upstream don't reopen this exact gap.
            safe=$(( ((need / 1024) + 1) * 1024 ))
            echo "   🔎 Known failure: Mamba cache align mode needs max-num-batched-tokens"
            echo "       >= $need — retrying with --max-num-batched-tokens $safe."
            _VLLM_ARG_OVERRIDE=(--max-num-batched-tokens "$safe")
            return 0
        fi
    fi
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# PRE-SERVE STACK PRE-FLIGHT
# Catches broken-environment failures ONCE, up front, instead of letting every
# selected model die identically with a 200-line EngineCore traceback in its own
# log file. Runs in --serve-only and headless too, since those are the normal
# restart paths and the ones most likely to inherit a half-finished pip upgrade.
#
# Three layers, cheap → specific:
#   1. `pip check` — surfaces ANY unsatisfied/conflicting dependency in the venv,
#      not just flashinfer. Advisory: vLLM venvs routinely carry benign warnings,
#      so this reports but never blocks.
#   2. flashinfer version reconciliation (self-healing — see above).
#   3. `import vllm` — the coarse gate.
# NOTE the flashinfer result is tracked separately from the vLLM import: `import
# vllm` succeeds even with a broken flashinfer, because vLLM only reaches for it
# later, inside Sampler.__init__ in the engine subprocess. Gating on the vLLM
# import alone would green-light exactly the serve that then dies.
# Returns non-zero when the environment cannot serve.
# Usage: _preflight_vllm_stack <venv_dir>
# ─────────────────────────────────────────────────────────────────────────────
_preflight_vllm_stack() {
    local venv="$1"
    local py="$venv/bin/python" pip="$venv/bin/pip"

    echo ""
    echo "--- Pre-flight: verifying vLLM environment at $venv ---"

    if [ ! -x "$py" ]; then
        echo "⚠️  No python at $py — skipping pre-flight (serve may still work via PATH)."
        return 0
    fi

    # 1. Advisory dependency-consistency scan.
    if [ -x "$pip" ]; then
        local pipchk
        pipchk=$("$pip" check 2>&1)
        if [ -n "$pipchk" ] && ! echo "$pipchk" | grep -qi "no broken requirements"; then
            echo "⚠️  pip reports dependency conflicts (advisory — not blocking):"
            echo "$pipchk" | head -10 | sed 's/^/   | /'
        else
            echo "✅ pip dependency check clean"
        fi
    fi

    # 2. Restore any package that has drifted off vLLM's exact pins, THEN reconcile
    #    flashinfer. Order matters: flashinfer's own resolution can pull siblings
    #    (nvidia-cutlass-dsl, apache-tvm-ffi) forward, so it runs last and gets the
    #    final say on its own packages.
    _align_vllm_pins "$venv"

    local fi_ok=1
    _fix_flashinfer_versions "$venv" || fi_ok=0

    # 3. GPU visibility — the most fundamental gate. Without a device vLLM cannot
    #    even build its CLI parser, so this must be checked regardless of how
    #    healthy the Python packages look.
    local gpu_ok=1
    _preflight_gpu "$venv" || gpu_ok=0
    # Record a working torch so a future clobber can be rolled back by version.
    [ "$gpu_ok" -eq 1 ] && _stamp_known_good_torch "$venv"

    # 4. Hard gates.
    if ! "$py" -c 'import vllm' >/dev/null 2>&1; then
        echo "❌ CRITICAL: vLLM cannot be imported — every model would fail identically."
        "$py" -c 'import vllm' 2>&1 | tail -15 | sed 's/^/   | /'
        echo "   Repair the venv before serving:  $pip install -U vllm"
        echo ""
        return 1
    fi
    echo "✅ vLLM imports cleanly: $("$py" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

    if [ "$fi_ok" -eq 0 ]; then
        echo "❌ CRITICAL: vLLM imports, but flashinfer does not — EngineCore will still"
        echo "   die at startup for every model. Resolve the flashinfer error above first."
        echo ""
        return 1
    fi

    if [ "$gpu_ok" -eq 0 ]; then
        echo "❌ CRITICAL: no usable GPU — vLLM would abort with 'Failed to infer device"
        echo "   type' while parsing arguments. Resolve the GPU error above first."
        echo ""
        return 1
    fi

    echo ""
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Select models to download  (skipped in --serve-only mode)
# ─────────────────────────────────────────────────────────────────────────────
DL_SELECTED=()
if [ "$SERVE_ONLY" -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  STEP 1 of 2 — Select models to DOWNLOAD"
    echo "════════════════════════════════════════════════════════════════════"
    _checkbox_menu "Available models (toggle with numbers, d=done):" "false" DL_SELECTED DEFAULT_DL_INDICES
else
    echo ""
    echo "  ⏭️  --serve-only: skipping download step (Step 1)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Select models to serve
# ─────────────────────────────────────────────────────────────────────────────
RUN_SELECTED=()
if [ "$HEADLESS" -eq 1 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  HEADLESS MODE (--start) — no prompts, serving requested models"
    echo "════════════════════════════════════════════════════════════════════"
    _resolve_start_specs "$START_SPECS"
else
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    if [ "$SERVE_ONLY" -eq 0 ]; then
        echo "  STEP 2 of 2 — Select models to SERVE with vLLM"
    else
        echo "  Select models to SERVE with vLLM"
    fi
    echo "  (ASR/NeMo models are download-only and excluded from this list)"
    echo "════════════════════════════════════════════════════════════════════"
    _checkbox_menu "Models to serve with vLLM (toggle with numbers, d=done):" "true" RUN_SELECTED DEFAULT_SERVE_INDICES

    # Offer to free a previous run's models before measuring available memory.
    _maybe_shutdown_existing_models

    _check_vram
fi

# Assign predictable sequential ports (BASE_PORT, +1, …) in launch order.
_assign_sequential_ports

echo ""
[ "$SERVE_ONLY" -eq 0 ] && echo "  Download : ${#DL_SELECTED[@]} model(s) selected"
echo "  Serve    : ${#RUN_SELECTED[@]} model(s) selected"
if [ "${#RUN_SELECTED[@]}" -gt 0 ]; then
    echo "  Port map (launch order, base ${BASE_PORT}):"
    _seq_n=1
    for _idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$_idx]}" = "0" ] && continue
        _pin=""
        [ "${MDL_PORT_EXPLICIT[$_idx]:-0}" = "1" ] && _pin="  (pinned)"
        printf "    %d. %-46s port %s%s\n" "$_seq_n" "${MDL_NAME[$_idx]}" "${MDL_PORT[$_idx]}" "$_pin"
        _seq_n=$((_seq_n + 1))
    done
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL IDLE-SLEEP TIMEOUT — prompt for each selected servable model
# Models offload weights to CPU after N idle minutes (auto-wake on next request).
# Default shown in brackets: catalog SLEEP_MIN override if set, else IDLE_SLEEP_MINUTES.
# ─────────────────────────────────────────────────────────────────────────────
_has_servable=0
for _idx in "${RUN_SELECTED[@]}"; do
    [ "${MDL_PORT[$_idx]}" = "0" ] && continue
    _has_servable=1; break
done

# Headless mode: never prompt — each model keeps its catalog SLEEP_MIN override,
# or falls back to the global IDLE_SLEEP_MINUTES (handled by the watchdog setup).
if [ "$_has_servable" = "1" ] && [ "$HEADLESS" -eq 0 ]; then
    echo "  ── Per-model idle-sleep timeout ─────────────────────────────────"
    echo "  Each model offloads weights to CPU after N idle minutes, then"
    echo "  auto-wakes on the next request.  Press Enter to accept the default."
    echo ""
    for _idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$_idx]}" = "0" ] && continue
        _default="${MDL_SLEEP[$_idx]:-$IDLE_SLEEP_MINUTES}"
        [ -z "$_default" ] && _default=0
        printf "  %-48s [default %s min, 0=never]: " "${MDL_NAME[$_idx]}" "$_default"
        read -r _sleep_input
        if [ -z "$_sleep_input" ]; then
            MDL_SLEEP[$_idx]="$_default"
        elif [[ "$_sleep_input" =~ ^[0-9]+$ ]]; then
            MDL_SLEEP[$_idx]="$_sleep_input"
        else
            echo "  ⚠️  Invalid — using default ${_default} min."
            MDL_SLEEP[$_idx]="$_default"
        fi
    done
    echo ""
    echo "  Sleep timeouts confirmed:"
    for _idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$_idx]}" = "0" ] && continue
        _mins="${MDL_SLEEP[$_idx]:-0}"
        if [ "$_mins" -eq 0 ] 2>/dev/null; then
            printf "    %-48s : never\n" "${MDL_NAME[$_idx]}"
        else
            printf "    %-48s : %d min\n" "${MDL_NAME[$_idx]}" "$_mins"
        fi
    done
    echo "  ─────────────────────────────────────────────────────────────────"
    echo ""
fi

if [ "$SERVE_ONLY" -eq 1 ]; then
    # ── Serve-only mode: skip all install/download steps ──────────────────────
    echo "⏭️  --serve-only: skipping apt, docker, venv, NeMo, HF, and download steps."
    echo ""

    # Find an existing venv so the VLLM_BIN search has a venv path to try first.
    VENV_DIR=""
    for candidate in "$VLLM_VENV" "$HOME/vllm-install/.vllm" "/home/cgray/vllm-install/.vllm"; do
        if [ -x "$candidate/bin/python" ]; then
            VENV_DIR="$candidate"
            echo "✅ Using existing venv at $VENV_DIR"
            break
        fi
    done
    [ -z "$VENV_DIR" ] && VENV_DIR="$VLLM_VENV"  # fallback; VLLM_BIN PATH search will cover it

    # FORCE_VLLM_REINSTALL previously only worked in full-install mode — which
    # --start/--serve-only skip entirely, so the flag was unreachable from the
    # normal restart command. Honor it here too.
    if [ "${FORCE_VLLM_REINSTALL:-false}" = "true" ]; then
        _rebuild_vllm_venv "" force || echo "⚠️  Forced venv rebuild failed — continuing with the existing venv."
    fi

else
    # ── Full install mode ──────────────────────────────────────────────────────
    sudo apt update
    sudo apt install -y --no-install-recommends wget curl gnupg2 git libgl1 libglib2.0-0
    sudo apt install -y jq
    sudo apt install -y python3.12-dev python3-dev build-essential ninja-build

    #-------- Docker / Containers ------------
    if command -v docker >/dev/null 2>&1; then
        echo "✅ Docker is installed. Version: $(docker --version)"
    else
        echo "❌ Docker is not installed."
        echo " You need docker first before running this. This will download a docker installer and run it for you. "
        wget -O "install_docker.sh" https://raw.githubusercontent.com/c2theg/srvBuilds/refs/heads/master/install_docker.sh
        chmod u+x install_docker.sh
        ./install_docker.sh
    fi

    #--- SETUP vLLM on DGX Spark ---
    # The vendor installer creates its venv at ./vllm-install/.vllm RELATIVE TO CWD,
    # and builds Triton from source (~10 min, ~1 GB). Run it from a new directory and
    # you get a second venv at a path the search below does not even look in — so the
    # whole build is discarded and the old venv is used anyway. Skip it whenever a
    # working vLLM venv already exists; force with FORCE_VLLM_REINSTALL=true.
    _existing_venv=""
    for candidate in "$VLLM_VENV" "$HOME/vllm-install/.vllm" "/home/cgray/vllm-install/.vllm" \
                     "$PWD/vllm-install/.vllm"; do
        if [ -x "$candidate/bin/python" ] && "$candidate/bin/python" -c "import vllm" 2>/dev/null; then
            _existing_venv="$candidate"
            break
        fi
    done

    if [ -n "$_existing_venv" ] && [ "${FORCE_VLLM_REINSTALL:-false}" != "true" ]; then
        echo "✅ Working vLLM venv already present at $_existing_venv — skipping the vendor installer."
        echo "   (It would rebuild Triton from source into $PWD/vllm-install and discard the result.)"
        echo "   Force a full reinstall with:  FORCE_VLLM_REINSTALL=true $0"
    else
        echo "--- Running the DGX Spark vLLM vendor installer (builds Triton, ~10 min) ---"
        echo "   Installing into: $PWD/vllm-install"
        curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
    fi

    VENV_PIP=""
    VENV_DIR=""
    # $PWD/vllm-install/.vllm is included so a venv the vendor installer just created
    # in the current directory is actually found, instead of being silently orphaned.
    for candidate in "$VLLM_VENV" "$HOME/vllm-install/.vllm" "/home/cgray/vllm-install/.vllm" \
                     "$PWD/vllm-install/.vllm"; do
        if [ -x "$candidate/bin/pip" ]; then
            VENV_PIP="$candidate/bin/pip"
            VENV_DIR="$candidate"
            echo "✅ Using vLLM venv at $candidate"
            break
        fi
    done

    if [ -z "$VENV_PIP" ]; then
        echo "⚠️  vLLM venv not found — creating dedicated downloader venv at $VLLM_VENV"
        python3 -m venv "$VLLM_VENV"
        VENV_PIP="$VLLM_VENV/bin/pip"
        VENV_DIR="$VLLM_VENV"
    fi

    if ! "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
        echo "⚠️  vllm not found in venv — installing via pip..."
        "$VENV_PIP" install -U vllm
        if "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
            echo "✅ vllm installed successfully"
        else
            echo "❌ vllm install failed — check pip output above"
        fi
        # Fresh install pulls flashinfer-python; make sure its companions match.
        _fix_flashinfer_versions "$VENV_DIR"
    else
        echo "✅ vllm already installed: $("$VENV_DIR/bin/python" -c 'import vllm; print(vllm.__version__)')"
    fi

    "$VENV_PIP" install -U "huggingface_hub[cli]" sentence-transformers

    if [ -x "$VENV_DIR/bin/hf" ]; then
        HF_CLI="$VENV_DIR/bin/hf"
        HF_LOGIN="$HF_CLI auth login"
    else
        HF_CLI="$VENV_DIR/bin/huggingface-cli"
        HF_LOGIN="$HF_CLI login"
    fi
    echo "✅ Using HF CLI: $HF_CLI"

    python3 -m venv "$NEMO_VENV"
    "$NEMO_VENV/bin/pip" install -U pip
    "$NEMO_VENV/bin/pip" install "nemo_toolkit[asr]"

    if [ -n "$HF_TOKEN" ]; then
        $HF_LOGIN --token "$HF_TOKEN"
        HF_AUTH="--token $HF_TOKEN"
    else
        echo "⚠️  HF_TOKEN not set — gated models will fail."
        HF_AUTH=""
    fi

    mkdir -p "$MODELS_DIR"
    HF_DL="$HF_CLI download $HF_AUTH"

    # ── Download selected models ───────────────────────────────────────────────
    if [ "${#DL_SELECTED[@]}" -eq 0 ]; then
        echo "⏭️  No models selected for download — skipping."
    else
        for idx in "${DL_SELECTED[@]}"; do
            echo ""
            echo "--- Downloading ${MDL_NAME[$idx]} ---"
            echo "    HF repo  : ${MDL_HF[$idx]}"
            echo "    Local dir: $MODELS_DIR/${MDL_DIR[$idx]}"
            if [ "${MDL_CAT[$idx]}" = "Super Large" ]; then
                echo "    ⚠️  SUPER LARGE model (~${MDL_DISK[$idx]} GB) — this will take a while."
                echo "    ℹ️  Nemotron-3-Super info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard"
            fi
            if $HF_DL "${MDL_HF[$idx]}" --local-dir "$MODELS_DIR/${MDL_DIR[$idx]}"; then
                echo "✅ ${MDL_NAME[$idx]} downloaded"
            else
                echo "❌ ${MDL_NAME[$idx]} download FAILED (see error above) — skipping."
            fi
        done
    fi

    echo ""
    echo "✅ All selected models downloaded to $MODELS_DIR"
fi  # end SERVE_ONLY check

# Upgrade vLLM if a newer release is out (best-effort; VENV_DIR is set by now in
# both full-install and --serve-only/headless modes).
_maybe_update_vllm

# ─────────────────────────────────────────────────────────────────────────────
# SERVE selected models with vLLM
# ─────────────────────────────────────────────────────────────────────────────
VLLM_LOGS="$BASE_DIR/logs"
mkdir -p "$VLLM_LOGS"
echo "✅ Log directory: $VLLM_LOGS"

VLLM_BIN=""
for candidate in \
    "$VENV_DIR/bin/vllm" \
    "$HOME/vllm-install/.vllm/bin/vllm" \
    "$HOME/.local/bin/vllm" \
    "/usr/local/bin/vllm" \
    "$(find "$HOME/vllm-install" -name vllm -type f 2>/dev/null | head -1)"; do
    if [ -x "$candidate" ]; then
        VLLM_BIN="$candidate"
        echo "✅ Found vllm binary at $VLLM_BIN"
        break
    fi
done

if [ -z "$VLLM_BIN" ]; then
    echo "⚠️  vllm not found in venv candidates — checking PATH..."
    if command -v vllm &>/dev/null; then
        VLLM_BIN="$(command -v vllm)"
        echo "✅ Found vllm on PATH: $VLLM_BIN"
    else
        echo "❌ CRITICAL: vllm binary not found anywhere."
        echo "   Searched venv: $VENV_DIR"
        echo "   Run: source $VENV_DIR/bin/activate && pip install -U vllm"
        echo "   Serve section will be skipped."
    fi
fi

# Verify + self-heal the venv before anything destructive happens. This sits ahead
# of the clean-start block on purpose: that block kills every running vLLM process,
# so proceeding with a venv that cannot serve would take down working models and
# leave nothing in their place. Aborting here leaves the current state untouched.
if [ -n "${VENV_DIR:-}" ]; then
    if ! _preflight_vllm_stack "$VENV_DIR"; then
        if [ "$PREFLIGHT_STRICT" = "true" ]; then
            echo "⛔ Pre-flight failed — aborting before the clean-start step."
            echo "   Nothing was killed; any models already running are still up."
            echo "   Fix the errors above and re-run, or serve anyway with:"
            echo "     PREFLIGHT_STRICT=false $0 $*"
            exit 1
        fi
        echo "⚠️  PREFLIGHT_STRICT=false — continuing despite a failed pre-flight."
    fi
fi

vllm_serve() {
    if [ -n "$VLLM_BIN" ]; then
        "$VLLM_BIN" serve "$@"
    else
        echo "⚠️  vllm not found — trying python module fallback"
        "$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server "$@"
    fi
}

# Print a failed model's log tail AND try to surface the real root cause.
# vLLM wraps startup errors as "Engine core initialization failed. See root
# cause above." — the actual exception is dozens of lines higher, so tail alone
# hides it. This shows the last N lines and greps the whole log for the earliest
# error/exception lines (usually the true cause).
_show_log_tail() {
    local log_file="$1" n="${2:-60}"
    tail -"$n" "$log_file" 2>/dev/null | sed 's/^/   | /'
    local rc
    # Primary: real raised-exception lines ("ClassName: message"), minus vLLM's
    # generic wrapper and traceback frames. The LAST such line is usually the true
    # cause — EngineCore raises it before the APIServer's "Engine core
    # initialization failed. See root cause above" wrapper. Matching the
    # "Error:/Exception:" signature also skips the huge INFO config dump (which
    # otherwise false-matches on substrings like device_config=cuda).
    rc=$(grep -aE '[A-Za-z_][A-Za-z0-9_.]*(Error|Exception): ' "$log_file" 2>/dev/null \
         | grep -avE 'File "|Engine core initialization failed|See root cause above' \
         | tail -3)
    # Fallback: broad phrase grep if no clean exception line was found.
    [ -z "$rc" ] && rc=$(grep -aiE 'out of memory|not supported|unsupported|no kernel|no such file|assertion|is not implemented|failed to' \
         "$log_file" 2>/dev/null | grep -avE 'INFO |DEBUG |File "' | tail -3)
    if [ -n "$rc" ]; then
        echo "   ── likely root cause (exception lines from the full log) ──"
        printf '%s\n' "$rc" | sed 's/^/   » /'
    fi
    # EngineCore logs its FULL traceback as ERROR lines tagged [core.py:NNN].
    # Those frames name the file that actually raised — e.g. WHICH package calls
    # a missing symbol — which the bare exception line never shows. Without this
    # we know WHAT is missing but not WHO wants it.
    local etb
    etb=$(grep -aE 'ERROR [0-9-]+ [0-9:]+ \[core\.py:[0-9]+\]' "$log_file" 2>/dev/null | tail -25)
    if [ -n "$etb" ]; then
        echo "   ── engine traceback (who raised it) ──"
        printf '%s\n' "$etb" | sed 's/^/   » /'
    fi
    echo "   → Full log: cat $log_file"
}

# Echo a usable HuggingFace CLI path (venv first, then PATH), or empty if none.
# Works in every mode: full-install sets HF_CLI, but --serve-only/headless skip
# that block, so we re-resolve here.
_find_hf_cli() {
    local c
    for c in "$VENV_DIR/bin/hf" "$VENV_DIR/bin/huggingface-cli" \
             "$HOME/vllm-install/.vllm/bin/hf" "$HOME/vllm-install/.vllm/bin/huggingface-cli" \
             "/home/cgray/vllm-install/.vllm/bin/hf" "/home/cgray/vllm-install/.vllm/bin/huggingface-cli"; do
        [ -x "$c" ] && { echo "$c"; return 0; }
    done
    command -v hf              >/dev/null 2>&1 && { command -v hf; return 0; }
    command -v huggingface-cli >/dev/null 2>&1 && { command -v huggingface-cli; return 0; }
    return 1
}

# Ensure catalog model <idx> is present on disk, downloading it if missing and
# AUTO_DOWNLOAD is on. Returns 0 if the model is (now) present, 1 otherwise.
# 'hf download' and 'huggingface-cli download' share the same subcommand/flags.
_ensure_model_downloaded() {
    local idx="$1"
    local name="${MDL_NAME[$idx]}" repo="${MDL_HF[$idx]}" dir="${MDL_DIR[$idx]}"
    local model_path="$MODELS_DIR/$dir"
    [ -f "$model_path/config.json" ] && return 0    # already downloaded

    if [ "${AUTO_DOWNLOAD:-true}" != "true" ]; then
        echo "  ℹ️  $name not on disk and AUTO_DOWNLOAD=false — not downloading."
        return 1
    fi

    local hf_cli
    hf_cli=$(_find_hf_cli) || {
        echo "  ❌ $name is missing and no HuggingFace CLI was found to download it."
        echo "     Install it:  $VENV_DIR/bin/pip install -U 'huggingface_hub[cli]'"
        return 1
    }

    echo ""
    echo "  ⬇️  $name not found locally — auto-downloading before serving:"
    echo "       repo : $repo"
    echo "       dest : $model_path"
    [ "${MDL_CAT[$idx]}" = "Super Large" ] && \
        echo "       ⚠️  SUPER LARGE (~${MDL_DISK[$idx]} GB) — this can take a long time."
    [ -z "$HF_TOKEN" ] && echo "       ⚠️  HF_TOKEN not set — this will fail if the repo is gated."

    local auth=""
    [ -n "$HF_TOKEN" ] && auth="--token $HF_TOKEN"
    mkdir -p "$MODELS_DIR"
    if "$hf_cli" download $auth "$repo" --local-dir "$model_path"; then
        if [ -f "$model_path/config.json" ]; then
            echo "  ✅ $name downloaded."
            return 0
        fi
        echo "  ⚠️  Download finished but no config.json under $model_path — check the repo id."
        return 1
    fi
    echo "  ❌ Auto-download failed for $repo (gated repo without HF_TOKEN, wrong id, or network)."
    return 1
}

# Pre-flight memory check. vLLM reserves (gpu-memory-utilization × total pool)
# and requires that much to be FREE at startup, else it dies with
# "Free memory ... is less than desired GPU memory utilization". This catches
# that early with an actionable message instead of a cryptic engine-core crash.
# Uses /proc/meminfo MemAvailable as the free-memory proxy (best-effort: it can
# be optimistic vs vLLM's CUDA view, so a pass isn't a guarantee — a fail is).
# Args: $1 = model name, $2 = the gpu-memory-utilization fraction.
_preflight_memory() {
    local name="$1" gmu="$2" mt_kb ma_kb
    mt_kb=$(awk '/^MemTotal:/{print $2}'     /proc/meminfo 2>/dev/null)
    ma_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)
    [[ "$mt_kb" =~ ^[0-9]+$ ]] && [[ "$ma_kb" =~ ^[0-9]+$ ]] || return 0  # can't read → don't block

    local total_gb=$(( mt_kb / 1024 / 1024 ))
    local avail_gb=$(( ma_kb / 1024 / 1024 ))
    local req_gb
    req_gb=$(awk -v t="$total_gb" -v f="$gmu" 'BEGIN{printf "%d", (t*f)+0.999}')  # ceil
    [ "$req_gb" -le "$avail_gb" ] && return 0

    echo ""
    echo "  ⚠️  Pre-flight: $name wants --gpu-memory-utilization $gmu ≈ ${req_gb} GB,"
    echo "      but only ${avail_gb} GB of ${total_gb} GB is free right now."
    echo "      vLLM needs the whole reservation free at startup, so it would fail with"
    echo "      a 'Free memory … is less than desired' ValueError."
    echo "      → Free memory (stop other models: pkill -9 -f 'vllm serve') or lower"
    echo "        this model's --gpu-memory-utilization."
    if [ "$HEADLESS" -eq 1 ]; then
        echo "      Headless: skipping $name to avoid a doomed launch."
        return 1
    fi
    printf "  Try to launch it anyway? [y/N]: "
    read -r _ans
    [[ "$_ans" =~ ^[Yy]$ ]] && return 0
    echo "      Skipping $name."
    return 1
}

# Helper: launch one model, echo command, wait 2s, confirm process is alive
# Usage: _vllm_launch <catalog_idx> [extra vllm args...]
_vllm_launch() {
    local idx="$1"; shift
    local name="${MDL_NAME[$idx]}"
    local dir="${MDL_DIR[$idx]}"
    local port="${MDL_PORT[$idx]}"
    local model_path="$MODELS_DIR/$dir"
    local log_file="$VLLM_LOGS/vllm-${port}.log"

    if [ ! -f "$model_path/config.json" ]; then
        # Not on disk yet — try to fetch it before giving up (AUTO_DOWNLOAD).
        if ! _ensure_model_downloaded "$idx"; then
            echo "⚠️  [idx $idx] $name — model not available at $model_path and could not be"
            echo "     downloaded. Set HF_TOKEN / AUTO_DOWNLOAD, or pre-download it, then retry."
            return 1
        fi
    fi

    # Predictable-port guard: make sure nothing else already holds this port.
    if ! _ensure_port_available "$port" "$name"; then
        echo "⚠️  [idx $idx] $name — port $port unavailable; not starting this model."
        return 1
    fi

    # Pre-flight memory check: pull the gpu-memory-utilization out of the args and
    # confirm the reservation fits in free RAM before we bother launching.
    local _a _prev="" _gmu=""
    for _a in "$@"; do
        [ "$_prev" = "--gpu-memory-utilization" ] && { _gmu="$_a"; break; }
        _prev="$_a"
    done
    if [ -n "$_gmu" ] && ! _preflight_memory "$name" "$_gmu"; then
        return 1
    fi

    local vllm_label
    if [ -n "$VLLM_BIN" ]; then
        vllm_label="$VLLM_BIN serve"
    else
        vllm_label="python3 -m vllm.entrypoints.openai.api_server"
    fi

    echo ""
    echo "--- Starting [idx $idx] $name on port $port ---"
    echo "    Model : $model_path"
    echo "    Log   : $log_file"
    echo "    CMD   : $vllm_label $model_path --host 0.0.0.0 --port $port --enable-sleep-mode $*"

    # Rotate any previous log. Headless mode skips the interactive clean-start
    # wipe, so this file otherwise accumulates across runs and the root-cause
    # extractor greps exceptions from OLD crashes (stale pids in the output).
    # The port-available guard above ensures nothing is still writing to it.
    [ -f "$log_file" ] && mv -f "$log_file" "${log_file}.old" 2>/dev/null

    vllm_serve "$model_path" --host 0.0.0.0 --port "$port" --enable-sleep-mode "$@" >> "$log_file" 2>&1 &
    local launch_pid=$!
    sleep 2
    if ! kill -0 "$launch_pid" 2>/dev/null; then
        echo "⚠️  $name (pid $launch_pid) exited immediately — log tail + root cause:"
        _show_log_tail "$log_file"
        _VLLM_ARG_OVERRIDE=()
        if [ "${_VLLM_RETRY:-0}" = "0" ] && _diagnose_and_repair "$log_file"; then
            echo "   🔁 Repair succeeded — retrying $name once…"
            mv -f "$log_file" "${log_file}.failed" 2>/dev/null
            local -a _override=("${_VLLM_ARG_OVERRIDE[@]+"${_VLLM_ARG_OVERRIDE[@]}"}")
            _VLLM_RETRY=1 _vllm_launch "$idx" "$@" "${_override[@]+"${_override[@]}"}"
            return $?
        fi
        if [ "${_VLLM_RETRY:-0}" != "2" ] && _rebuild_vllm_venv "$log_file"; then
            echo "   🔁 Venv rebuilt — final retry of $name…"
            mv -f "$log_file" "${log_file}.failed" 2>/dev/null
            _VLLM_RETRY=2 _vllm_launch "$idx" "$@"
            return $?
        fi
        return 1
    fi

    echo "✅ $name launched  pid=$launch_pid  port=$port"
    echo "   → Log: tail -f $log_file"
    echo "   ⏳ Waiting for model to finish loading before starting the next one..."

    # Poll every 5s (was 15s) so small models are detected ready up to ~14s
    # sooner; print progress only every 30s so the log stays as quiet as before.
    # /health is cheaper than /v1/models and returns 200 once the engine is up.
    # First NVFP4/FP8 launch includes a one-time CUTLASS kernel compile, so the
    # timeout is configurable (VLLM_READY_TIMEOUT, default 1800s / 30 min).
    local elapsed=0 timeout="${VLLM_READY_TIMEOUT:-1800}" poll=5
    while true; do
        if ! kill -0 "$launch_pid" 2>/dev/null; then
            echo "   ❌ $name process died during loading — log tail + root cause:"
            _show_log_tail "$log_file"
            # Match the crash against the known-failure playbook; if a repair
            # applies and succeeds, retry ONCE. The old log is moved aside so the
            # retry's diagnosis can't re-match stale lines from this failure.
            _VLLM_ARG_OVERRIDE=()
            if [ "${_VLLM_RETRY:-0}" = "0" ] && _diagnose_and_repair "$log_file"; then
                echo "   🔁 Repair succeeded — retrying $name once…"
                mv -f "$log_file" "${log_file}.failed" 2>/dev/null
                local -a _override=("${_VLLM_ARG_OVERRIDE[@]+"${_VLLM_ARG_OVERRIDE[@]}"}")
                _VLLM_RETRY=1 _vllm_launch "$idx" "$@" "${_override[@]+"${_override[@]}"}"
                return $?
            fi
            # Targeted repair already had its shot (or nothing matched). Final
            # escalation: fresh venv from the vendor installer, then one last try.
            if [ "${_VLLM_RETRY:-0}" != "2" ] && _rebuild_vllm_venv "$log_file"; then
                echo "   🔁 Venv rebuilt — final retry of $name…"
                mv -f "$log_file" "${log_file}.failed" 2>/dev/null
                _VLLM_RETRY=2 _vllm_launch "$idx" "$@"
                return $?
            fi
            return 1
        fi
        if curl -sf --max-time 5 "http://localhost:${port}/health" > /dev/null 2>&1 || \
           curl -sf --max-time 5 "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "   ✅ $name ready on port $port  (${elapsed}s)"
            echo "   → Status: curl -s http://localhost:${port}/v1/models | jq ."
            return 0
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "   ⚠️  $name not ready after ${timeout}s — moving on; check: tail -f $log_file"
            return 0
        fi
        sleep "$poll"
        elapsed=$((elapsed + poll))
        [ $((elapsed % 30)) -eq 0 ] && printf "   [%ds] still loading...\n" "$elapsed"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP WATCHDOG — polls vLLM /metrics and offloads idle models to CPU
# Args: one "port:idle_seconds" pair per model to watch. The idle threshold is
#       per-port, so models can carry different sleep timeouts (catalog SLEEP_MIN).
# ─────────────────────────────────────────────────────────────────────────────

# Build _WATCH_PAIRS ("port:idle_seconds") from RUN_SELECTED. Each model uses its
# MDL_SLEEP value (catalog override or the runtime-prompted value), falling back
# to the global IDLE_SLEEP_MINUTES; models with an effective 0 are skipped.
_build_watch_pairs() {
    _WATCH_PAIRS=()
    local _idx _p _mins
    for _idx in "${RUN_SELECTED[@]}"; do
        _p="${MDL_PORT[$_idx]}"
        [ "$_p" = "0" ] && continue
        _mins="${MDL_SLEEP[$_idx]:-}"
        [ -z "$_mins" ] && _mins="$IDLE_SLEEP_MINUTES"
        [ "$_mins" -gt 0 ] 2>/dev/null || continue
        _WATCH_PAIRS+=("${_p}:$((_mins * 60))")
    done
}

_start_sleep_watchdog() {
    local -a watch_pairs=("$@")
    [ "${#watch_pairs[@]}" -eq 0 ] && return 0

    # Build the port list and the port→idle-seconds map from the pairs.
    local ports_str="" idle_map=""
    for pair in "${watch_pairs[@]}"; do
        local p="${pair%%:*}" s="${pair##*:}"
        ports_str="${ports_str}${ports_str:+ }${p}"
        idle_map="${idle_map}${idle_map:+ }[${p}]=${s}"
    done

    local watchdog_script="$VLLM_LOGS/sleep_watchdog.sh"
    cat > "$watchdog_script" << 'WATCHDOG_EOF'
#!/usr/bin/env bash
# vLLM sleep watchdog — generated by install_ai_spark_vllm.sh
# Per-port idle threshold (seconds) — a model sleeps once idle past its own value.
declare -A IDLE_SECS_BY_PORT=( __IDLE_MAP__ )
PORTS=(__PORTS__)
declare -A last_request_count
declare -A last_active

for port in "${PORTS[@]}"; do
    cnt=$(curl -sf --max-time 5 "http://localhost:${port}/metrics" 2>/dev/null \
        | awk '/^vllm:e2e_request_latency_seconds_count/{sum+=$2} END{print int(sum+0)}')
    last_request_count[$port]="${cnt:-0}"
    last_active[$port]=$(date +%s)
done

while true; do
    sleep 60
    now=$(date +%s)
    for port in "${PORTS[@]}"; do
        curl -sf --max-time 3 "http://localhost:${port}/health" > /dev/null 2>&1 || continue
        new_cnt=$(curl -sf --max-time 5 "http://localhost:${port}/metrics" 2>/dev/null \
            | awk '/^vllm:e2e_request_latency_seconds_count/{sum+=$2} END{print int(sum+0)}')
        new_cnt="${new_cnt:-0}"
        idle_secs="${IDLE_SECS_BY_PORT[$port]:-900}"
        if [ "$new_cnt" != "${last_request_count[$port]:-0}" ]; then
            last_request_count[$port]="$new_cnt"
            last_active[$port]=$now
        else
            idle=$(( now - ${last_active[$port]:-$now} ))
            if [ "$idle" -ge "$idle_secs" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] port ${port}: idle $((idle/60))m (threshold $((idle_secs/60))m) — sleeping (offloading to CPU)"
                # Try /v1/sleep first (vLLM ≥0.20.x), fall back to /sleep (older)
                if curl -sf -X POST "http://localhost:${port}/v1/sleep" > /dev/null 2>&1 || \
                   curl -sf -X POST "http://localhost:${port}/sleep"    > /dev/null 2>&1; then
                    last_active[$port]=$now
                fi
            fi
        fi
    done
done
WATCHDOG_EOF

    sed -i \
        -e "s|__IDLE_MAP__|${idle_map}|g" \
        -e "s|__PORTS__|${ports_str}|g" \
        "$watchdog_script"
    chmod +x "$watchdog_script"

    nohup "$watchdog_script" >> "$VLLM_LOGS/sleep_watchdog.log" 2>&1 &
    local wpid=$!
    echo "✅ Sleep watchdog started  pid=$wpid"
    echo "   Per-port idle thresholds (min):"
    for pair in "${watch_pairs[@]}"; do
        printf "     port %-6s : %d min\n" "${pair%%:*}" "$(( ${pair##*:} / 60 ))"
    done
    echo "   Log: tail -f $VLLM_LOGS/sleep_watchdog.log"
    echo "   Sleep API: POST http://localhost:<port>/sleep  |  POST .../wake_up"
}

# ─────────────────────────────────────────────────────────────────────────────
# SQLITE STRUCTURED MEMORY — exact fact storage, 2-month / 100 MB retention
# ─────────────────────────────────────────────────────────────────────────────
_setup_sqlite_memory() {
    mkdir -p "$MEMORY_DIR"
    if ! command -v sqlite3 &>/dev/null; then
        echo "⚠️  sqlite3 not found — installing..."
        sudo apt install -y sqlite3
    fi

    sqlite3 "$SQLITE_DB" << 'SQL_EOF'
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    category    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT,
    confidence  REAL DEFAULT 1.0,
    tags        TEXT,
    UNIQUE(category, key)
);
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    model       TEXT,
    tokens_used INTEGER
);
CREATE INDEX IF NOT EXISTS idx_facts_created  ON facts(created_at);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_conv_created   ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conv_session   ON conversations(session_id);
SQL_EOF

    echo "✅ SQLite structured memory initialized: $SQLITE_DB"

    cat > "$MEMORY_DIR/sqlite_maintain.sh" << MAINT_EOF
#!/usr/bin/env bash
DB="$SQLITE_DB"
MAX_MB=$SQLITE_MAX_MB
RETENTION_DAYS=$SQLITE_RETENTION_DAYS

sqlite3 "\$DB" "DELETE FROM facts         WHERE created_at < datetime('now', '-\${RETENTION_DAYS} days');"
sqlite3 "\$DB" "DELETE FROM conversations WHERE created_at < datetime('now', '-\${RETENTION_DAYS} days');"
sqlite3 "\$DB" "VACUUM;"

SIZE_MB=\$(du -sm "\$DB" 2>/dev/null | cut -f1)
while [ "\${SIZE_MB:-0}" -gt "\$MAX_MB" ]; do
    sqlite3 "\$DB" "DELETE FROM facts         WHERE id IN (SELECT id FROM facts         ORDER BY created_at ASC LIMIT 500);"
    sqlite3 "\$DB" "DELETE FROM conversations WHERE id IN (SELECT id FROM conversations ORDER BY created_at ASC LIMIT 500);"
    sqlite3 "\$DB" "VACUUM;"
    SIZE_MB=\$(du -sm "\$DB" 2>/dev/null | cut -f1)
done
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] SQLite maintenance done. Size: \${SIZE_MB:-?}MB / ${SQLITE_MAX_MB}MB max"
MAINT_EOF

    chmod +x "$MEMORY_DIR/sqlite_maintain.sh"
    (crontab -l 2>/dev/null | grep -v "sqlite_maintain"; \
     echo "0 3 * * * $MEMORY_DIR/sqlite_maintain.sh >> $VLLM_LOGS/sqlite_maintain.log 2>&1") | crontab -
    echo "✅ SQLite retention: ${SQLITE_RETENTION_DAYS} days / ${SQLITE_MAX_MB} MB — daily cleanup at 3am"
    echo "   DB path : $SQLITE_DB"
    echo "   Maintain: $MEMORY_DIR/sqlite_maintain.sh"
}

# ─────────────────────────────────────────────────────────────────────────────
# QDRANT VECTOR MEMORY — semantic memory, 2-month / 1 GB retention
# ─────────────────────────────────────────────────────────────────────────────
_setup_qdrant() {
    mkdir -p "$QDRANT_DATA_DIR/storage" "$QDRANT_DATA_DIR/config"

    cat > "$QDRANT_DATA_DIR/config/production.yaml" << 'QDRANT_CFG_EOF'
storage:
  storage_path: /qdrant/storage
service:
  http_port: 6333
  grpc_port: 6334
QDRANT_CFG_EOF

    docker pull qdrant/qdrant:latest
    docker run -d \
        --name qdrant \
        --network host \
        -v "$QDRANT_DATA_DIR/storage:/qdrant/storage:rw" \
        -v "$QDRANT_DATA_DIR/config:/qdrant/config:ro" \
        qdrant/qdrant:latest

    echo "✅ Qdrant vector DB started"
    echo "   HTTP API : http://localhost:${QDRANT_HTTP_PORT}"
    echo "   gRPC     : localhost:${QDRANT_GRPC_PORT}"
    echo "   Dashboard: http://localhost:${QDRANT_HTTP_PORT}/dashboard"

    cat > "$QDRANT_DATA_DIR/qdrant_maintain.sh" << QDRANT_MAINT_EOF
#!/usr/bin/env bash
QDRANT_URL="http://localhost:${QDRANT_HTTP_PORT}"
MAX_GB=$QDRANT_MAX_GB
DATA_DIR="$QDRANT_DATA_DIR/storage"
CUTOFF=\$(date -d "-${QDRANT_RETENTION_DAYS} days" +%s 2>/dev/null || \
          date -v -${QDRANT_RETENTION_DAYS}d     +%s 2>/dev/null)

for col in \$(curl -sf "\$QDRANT_URL/collections" 2>/dev/null \
             | jq -r '.result.collections[].name' 2>/dev/null); do
    curl -sf -X POST "\$QDRANT_URL/collections/\${col}/points/delete" \
        -H "Content-Type: application/json" \
        -d "{\"filter\":{\"must\":[{\"key\":\"created_at\",\"range\":{\"lt\":\$CUTOFF}}]}}" > /dev/null
done

SIZE_BYTES=\$(du -sb "\$DATA_DIR" 2>/dev/null | cut -f1)
SIZE_GB=\$(( \${SIZE_BYTES:-0} / 1073741824 ))
if [ "\${SIZE_GB:-0}" -gt "\$MAX_GB" ]; then
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Qdrant storage \${SIZE_GB}GB exceeds \${MAX_GB}GB limit"
    echo "   Prune collections manually or reduce QDRANT_RETENTION_DAYS."
fi
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Qdrant maintenance done. Storage: \${SIZE_GB:-?}GB / ${QDRANT_MAX_GB}GB max"
QDRANT_MAINT_EOF

    chmod +x "$QDRANT_DATA_DIR/qdrant_maintain.sh"
    (crontab -l 2>/dev/null | grep -v "qdrant_maintain"; \
     echo "0 4 * * * $QDRANT_DATA_DIR/qdrant_maintain.sh >> $VLLM_LOGS/qdrant_maintain.log 2>&1") | crontab -
    echo "✅ Qdrant retention: ${QDRANT_RETENTION_DAYS} days / ${QDRANT_MAX_GB} GB — daily cleanup at 4am"
    echo "   Note: store 'created_at' (Unix epoch) as a payload field in each point for age-based pruning."
    echo "   Maintain: $QDRANT_DATA_DIR/qdrant_maintain.sh"
}

# Headless mode is additive: it must not kill models/containers another run (or
# another cron entry) already started, so the clean start only runs interactively.
if [ "$HEADLESS" -eq 0 ]; then
    echo "--- Clean start: killing all vLLM processes and removing old logs ---"
    docker stop open-webui searxng qdrant 2>/dev/null || true
    docker rm   open-webui searxng qdrant 2>/dev/null || true
    _kill_vllm_processes
    sleep 3
    rm -f "$VLLM_LOGS"/vllm-*.log
    echo "✅ Old vLLM processes killed and logs cleared"
fi

export TORCH_FLOAT32_MATMUL_PRECISION=high

# ── CUDA toolchain + JIT-compile parallelism (for NVFP4/FP8 kernel builds) ────
# FlashInfer compiles CUTLASS kernels with nvcc on first launch. Make sure nvcc
# is discoverable and cap the parallel compile jobs so the build doesn't OOM.
if [ -z "${CUDA_HOME:-}" ]; then
    for _cuda in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
        [ -x "$_cuda/bin/nvcc" ] && { export CUDA_HOME="$_cuda"; break; }
    done
fi
if [ -n "${CUDA_HOME:-}" ]; then
    case ":$PATH:" in *":$CUDA_HOME/bin:"*) : ;; *) export PATH="$CUDA_HOME/bin:$PATH" ;; esac
    echo "✅ CUDA toolchain: CUDA_HOME=$CUDA_HOME  ($("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release //p'))"
elif command -v nvcc >/dev/null 2>&1; then
    echo "✅ nvcc found on PATH: $(command -v nvcc)"
else
    echo "⚠️  nvcc not found — NVFP4/FP8 models that JIT-compile CUTLASS kernels may"
    echo "    fail. Install the CUDA toolkit or set CUDA_HOME to your CUDA install."
fi
# Cap FlashInfer/ninja compile parallelism to avoid OOM-killed ('Killed') builds.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
echo "✅ Kernel-compile parallelism: MAX_JOBS=$MAX_JOBS (lower to 2/1 if a build is OOM-killed)"

if [ "${#RUN_SELECTED[@]}" -eq 0 ]; then
    echo "⏭️  No models selected to serve — skipping vLLM startup."
else
    echo ""
    echo "  ── Models queued to serve ───────────────────────────────────────"
    for idx in "${RUN_SELECTED[@]}"; do
        [ "${MDL_PORT[$idx]}" = "0" ] && continue
        printf "    [catalog idx %2d]  %-50s  port %s\n" \
            "$idx" "${MDL_NAME[$idx]}" "${MDL_PORT[$idx]}"
    done
    echo "  ─────────────────────────────────────────────────────────────────"
fi

# ─────────────────────────────────────────────────────────────────────────────
# --gpu-memory-utilization on unified memory (DGX Spark / GB10, ~121 GB shared):
# vLLM's KV-cache profiler measures TOTAL system GPU memory in use as the
# baseline — this includes weights from ALL other running vLLM processes, not
# just this one. So when a 35B model is already loaded (~38 GB), even a tiny
# model needs a budget > (38 + its own weights) GB, i.e. utilization > 0.38.
# That is why embedding/reranker models use 0.45-0.50 here despite being small:
# the fraction buys enough headroom over the 35B's footprint to allow KV cache
# allocation. These fractions "over-subscribe" the pool on paper but in practice
# only 1-2 models are hot at a time (sleep watchdog offloads the rest to CPU).
#   Rough guide on a 121 GB box:  0.40 ≈ 48 GB,  0.50 ≈ 61 GB,  0.75 ≈ 91 GB.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PER-MODEL SERVE ARGS — _serve_model dispatches on the HF REPO ID, not the
# catalog index, so inserting/reordering _add lines can never mis-map a model
# to the wrong serve block (the v0.1.5 index-mismatch bug class is gone).
# A servable model with no dedicated entry falls through to safe generic
# defaults in the * arm. ASR/NeMo download-only models (PORT=0) are skipped.
#   NeMo ASR usage: python3 -c "
#     import nemo.collections.asr as nemo_asr
#     model = nemo_asr.models.EncDecRNNTBPEModel.restore_from('$MODELS_DIR/parakeet-tdt-0.6b-v3/model.nemo')
#     print(model.transcribe(['your_audio.wav']))"
# ─────────────────────────────────────────────────────────────────────────────
_serve_model() {
    local idx="$1"
    [ "${MDL_PORT[$idx]}" = "0" ] && return 0

    case "${MDL_HF[$idx]}" in

    "Qwen/Qwen3.6-35B-A3B-FP8")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-35B-A3B" \
            --dtype auto \
            --gpu-memory-utilization 0.40 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # Full DGX Spark profile (v0.3.26) — replaces the earlier 0.30-gmu/32768-
    # context/hermes-parser profile. No --quantization flag: vLLM auto-detects
    # NVFP4 from the checkpoint config, and --moe-backend marlin needs that
    # auto-detected path (an explicit --quantization modelopt_fp4 alongside it
    # was observed to conflict). --async-scheduling overlaps CPU scheduling with
    # GPU execution; --speculative-config enables MTP self-speculative decoding
    # (3 draft tokens/step, its own triton MoE backend, separate from the main
    # model's marlin backend). --load-format fastsafetensors is the faster
    # DGX Spark loader. 0.4 gmu is sized for the full 262144 context's KV cache
    # at fp8 — no longer the 0.30 co-run-with-27B profile (see the catalog
    # comment above), so check free memory before also starting the 27B model.
    "nvidia/Qwen3.6-35B-A3B-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-35B-A3B-NVFP4" \
            --tensor-parallel-size 1 \
            --trust-remote-code \
            --kv-cache-dtype fp8 \
            --attention-backend flashinfer \
            --moe-backend marlin \
            --gpu-memory-utilization 0.4 \
            --max-model-len 262144 \
            --max-num-seqs 4 \
            --max-num-batched-tokens 8192 \
            --enable-chunked-prefill \
            --async-scheduling \
            --enable-prefix-caching \
            --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
            --load-format fastsafetensors \
            --reasoning-parser qwen3 \
            --tool-call-parser qwen3_xml \
            --enable-auto-tool-choice
        ;;

    # Current HF card (nvidia-modelopt v0.45.0 / NVFP4 1.0) recommends:
    #   vllm serve nvidia/Qwen3.6-27B-NVFP4 --quantization modelopt
    #     --max-model-len 262144 --reasoning-parser qwen3
    # DGX Spark co-run default keeps max-model-len lower so this can live beside
    # the 35B-A3B NVFP4 process. For full solo context:
    #   QWEN36_27B_MAX_MODEL_LEN=262144 ./install_ai_spark_vllm.sh --start Qwen3.6-27B-NVFP4
    "nvidia/Qwen3.6-27B-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-27B-NVFP4" \
            --dtype auto \
            --quantization modelopt \
            --gpu-memory-utilization 0.25 \
            --max-model-len "${QWEN36_27B_MAX_MODEL_LEN:-32768}" \
            --kv-cache-dtype fp8 \
            --max-num-seqs 4 \
            --max-num-batched-tokens 8192 \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --trust-remote-code \
            --reasoning-parser qwen3
        ;;

    # Unsloth's "Fast" repack of the same NVFP4 checkpoint — same footprint.
    "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.6-35B-A3B-NVFP4-Fast" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.30 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # Nemotron-3-Nano is a hybrid Mamba/attention model — see the comment above
    # the Nemotron-Omni block for why --max-num-batched-tokens is required
    # whenever --enable-prefix-caching is on (Mamba cache "align" mode).
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Nano-30B-NVFP4" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.20 \
            --max-model-len 32768 \
            --max-num-batched-tokens 4096 \
            --max-num-seqs 178 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-Coder-30B" \
            --dtype auto \
            --gpu-memory-utilization 0.62 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        _vllm_launch "$idx" \
            --served-model-name "DeepSeek-R1-Distill-Qwen-32B" \
            --dtype auto \
            --gpu-memory-utilization 0.65 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    "google/gemma-4-31B-it")
        _vllm_launch "$idx" \
            --served-model-name "gemma-4-31B" \
            --dtype auto \
            --gpu-memory-utilization 0.60 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "nvidia/Gemma-4-31B-IT-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "gemma4-31b" \
            --trust-remote-code \
            --quantization modelopt \
            --tensor-parallel-size 1 \
            --language-model-only \
            --gpu-memory-utilization 0.46 \
            --max-model-len 32768 \
            --max-num-seqs 1 \
            --kv-cache-dtype fp8 \
            --calculate-kv-scales \
            --enable-prefix-caching \
            --enable-chunked-prefill
        ;;

    "google/gemma-4-26B-A4B-it")
        _vllm_launch "$idx" \
            --served-model-name "gemma-4-26B-A4B" \
            --dtype auto \
            --gpu-memory-utilization 0.55 \
            --max-model-len 16384 \
            --max-num-batched-tokens 4096 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # All three Omni variants below are hybrid Mamba/attention models. With
    # --enable-prefix-caching on, vLLM runs Mamba cache in "align" mode, which
    # computes a block_size from the model's mamba state layout (2128 for this
    # model at 32768 context) and then REQUIRES max_num_batched_tokens >= that
    # block_size. Without an explicit --max-num-batched-tokens vLLM's own default
    # (2048) is smaller, so the engine dies at KV-cache init:
    #   AssertionError: In Mamba cache align mode, block_size (2128) must be
    #   <= max_num_batched_tokens (2048)
    # 4096 clears the observed 2128 with headroom. See also _diagnose_and_repair,
    # which auto-retries any model that hits this with a computed safe value.
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Nano-Omni-30B-A3B" \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.62 \
            --max-model-len 32768 \
            --max-num-batched-tokens 4096 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    # FP8 (modelopt) build of the Omni reasoning model above — ~half the BF16
    # footprint. 0.35 × 121 ≈ 42 GB budget leaves headroom for the vision tower
    # + KV. vLLM auto-detects modelopt FP8 from the checkpoint config; the explicit
    # flag makes it deterministic — drop it if your vLLM build errors on it.
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8")
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Nano-Omni-30B-A3B-FP8" \
            --dtype auto \
            --quantization modelopt \
            --gpu-memory-utilization 0.35 \
            --max-model-len 32768 \
            --max-num-batched-tokens 4096 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    # NVFP4 (~4-bit) build — ~a quarter of the BF16 footprint. 0.20 × 121 ≈ 24 GB
    # budget. Same modelopt_fp4 path as the other Nemotron/Qwen NVFP4 entries.
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4")
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Nano-Omni-30B-A3B-NVFP4" \
            --dtype auto \
            --quantization modelopt_fp4 \
            --gpu-memory-utilization 0.20 \
            --max-model-len 32768 \
            --max-num-batched-tokens 4096 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    # --gpu-memory-utilization is a FRACTION OF THE WHOLE ~121.7 GB pool that THIS
    # model reserves for weights + KV, and vLLM requires that much to be FREE at
    # startup. So size it to the model's real footprint, NOT high: a big fraction
    # on a tiny model both wastes the pool and fails to start whenever memory is
    # tight (0.55 on the 4B reranker demanded 67 GB → the ValueError we hit).
    # Rough guide: 0.05≈6 GB, 0.10≈12 GB, 0.15≈18 GB, 0.20≈24 GB.
    # max-model-len capped at 8192 — embedding models default to 40960 (huge KV).
    "BAAI/bge-m3")   # ~2 GB weights
        _vllm_launch "$idx" \
            --served-model-name "bge-m3" \
            --dtype auto \
            --gpu-memory-utilization 0.06 \
            --max-model-len 8192 \
            --trust-remote-code
        ;;

    "Qwen/Qwen3-Embedding-4B")   # ~8 GB weights
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-Embedding-4B" \
            --dtype auto \
            --gpu-memory-utilization 0.10 \
            --max-model-len 8192 \
            --trust-remote-code
        ;;

    "BAAI/bge-reranker-v2-m3")   # ~2 GB weights
        _vllm_launch "$idx" \
            --served-model-name "bge-reranker-v2-m3" \
            --dtype auto \
            --gpu-memory-utilization 0.06 \
            --max-model-len 8192 \
            --trust-remote-code
        ;;

    # Generative yes/no reranker — clients score via logprobs on "yes"/"no"
    # tokens: POST /v1/completions with logprobs=1; compare P("yes") vs P("no").
    # ~9 GB weights; 0.12≈15 GB leaves room for KV at max-model-len 10000.
    "Qwen/Qwen3-Reranker-4B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-Reranker-4B" \
            --dtype auto \
            --gpu-memory-utilization 0.12 \
            --max-model-len 10000 \
            --enable-prefix-caching \
            --max-logprobs 20 \
            --trust-remote-code
        ;;

    # ── SUPER LARGE (120B+): need nearly the whole GPU — don't co-run others ──
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
        echo "   ℹ️  Model info: https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard"
        echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
        _vllm_launch "$idx" \
            --served-model-name "Nemotron-3-Super-120B-A12B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 8192 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "Qwen/Qwen3.5-122B-A10B")
        echo "   ⚠️  SUPER LARGE — needs ~120 GB VRAM. Ensure no other large models are running."
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-122B-A10B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 8192 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # FP8 uses ~62 GB VRAM vs ~120 GB for BF16 — fits easily, longer context.
    "Qwen/Qwen3.5-122B-A10B-FP8")
        echo "   ★  FP8 version: ~62 GB VRAM, faster inference, longer context than BF16"
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-122B-A10B-FP8" \
            --dtype auto \
            --gpu-memory-utilization 0.75 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "openai/gpt-oss-120b")
        echo "   ⚠️  SUPER LARGE — needs ~115 GB VRAM. Ensure no other large models are running."
        _vllm_launch "$idx" \
            --served-model-name "GPT-OSS-120B" \
            --dtype auto \
            --gpu-memory-utilization 0.93 \
            --max-model-len 8192 \
            --enable-prefix-caching \
            --trust-remote-code
        ;;

    # DGX-Spark-optimized profile from the model card (the plain "vllm serve"
    # profile it also lists omits the three VLLM_* env vars, --max-num-batched-
    # tokens/--max-num-seqs, and the tool/reasoning parsers — this is the fuller
    # one). --load-format fastsafetensors is the card's recommended loader
    # (~90s load on Spark). The three VLLM_* vars pin the Marlin NVFP4/FP8 GEMM
    # backend the card benchmarked ~2% faster than CUTLASS on GB10 (~17.5 vs
    # ~17.1 tok/s) — exported only for this launch, not globally. At 0.7
    # gpu-memory-utilization the card reports KV cache alone uses ~6.9 GB,
    # enough for 150K tokens at 2.1x concurrency up to the full 262144 context.
    # Card also notes a small accuracy trade-off from the NVFP4 quant: GSM8k
    # 88.2 → 85.8 (97.0% recovery vs. the unquantized model).
    "sjug/Qwen3.5-122B-A10B-NVFP4-resharded")
        echo "   ⚠️  SUPER LARGE — needs ~72 GB VRAM. Ensure no other large models are running."
        VLLM_NVFP4_GEMM_BACKEND=marlin \
        VLLM_TEST_FORCE_FP8_MARLIN=1 \
        VLLM_MARLIN_USE_ATOMIC_ADD=1 \
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-122B-A10B-NVFP4" \
            --load-format fastsafetensors \
            --kv-cache-dtype fp8 \
            --gpu-memory-utilization 0.7 \
            --max-model-len 262144 \
            --max-num-batched-tokens 8192 \
            --max-num-seqs 10 \
            --enable-prefix-caching \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            --reasoning-parser qwen3 \
            --trust-remote-code
        ;;

    # ── Small models (1-hour idle-sleep via catalog SLEEP_MIN=60) ─────────────
    # Fractions sized to each model's own footprint (weights + KV), not the pool.
    "Qwen/Qwen3.5-4B")   # ~8 GB weights; 0.12≈15 GB
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-4B" \
            --dtype auto \
            --gpu-memory-utilization 0.12 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    "Qwen/Qwen3.5-2B")   # ~4 GB weights; 0.08≈10 GB
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-2B" \
            --dtype auto \
            --gpu-memory-utilization 0.08 \
            --max-model-len 32768 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # ~18 GB weights; 0.20≈24 GB leaves ~6 GB for KV at max-model-len 16384.
    "Qwen/Qwen3.5-9B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3.5-9B" \
            --dtype auto \
            --gpu-memory-utilization 0.20 \
            --max-model-len 16384 \
            --enable-prefix-caching \
            --trust-remote-code \
            --enable-auto-tool-choice \
            --tool-call-parser hermes
        ;;

    # Served via vLLM's transcription task — POST /v1/audio/transcriptions.
    # STT endpoint, not chat: skipped in OpenWebUI chat auto-registration; wire
    # it in under Admin Settings → Audio → STT (URL http://localhost:8014/v1).
    "Qwen/Qwen3-ASR-1.7B")
        _vllm_launch "$idx" \
            --served-model-name "Qwen3-ASR-1.7B" \
            --dtype auto \
            --gpu-memory-utilization 0.07 \
            --trust-remote-code
        ;;

    *)  # No dedicated serve entry — conservative generic defaults.
        echo "   ℹ️  ${MDL_HF[$idx]} has no dedicated serve entry — using generic defaults."
        _vllm_launch "$idx" \
            --served-model-name "${MDL_DIR[$idx]}" \
            --dtype auto \
            --gpu-memory-utilization 0.50 \
            --max-model-len 16384 \
            --trust-remote-code
        ;;
    esac
}

# ── Launch every selected servable model, one at a time ───────────────────────
for idx in "${RUN_SELECTED[@]}"; do
    _serve_model "$idx"
done

# ── Headless mode ends here: start the watchdog, show status, and exit ────────
# (containers, OpenWebUI registration, and memory setup are interactive-run-only)
if [ "$HEADLESS" -eq 1 ]; then
    _build_watch_pairs
    if [ "${#_WATCH_PAIRS[@]}" -gt 0 ]; then
        echo "--- Starting vLLM sleep watchdog ---"
        _start_sleep_watchdog "${_WATCH_PAIRS[@]}"
    fi
    _show_vllm_status "FINAL"
    _verify_served_models
    echo "✅ Headless start complete."
    exit 0
fi

#---------------------------------------------------------------------------------------------------------------
#--- SearXNG (web search backend for OpenWebUI) ---
if [ "$ENABLE_SEARXNG" = "true" ]; then
    echo "--- Starting SearXNG container ---"
    mkdir -p "$BASE_DIR/searxng"

    if [ ! -f "$BASE_DIR/searxng/settings.yml" ]; then
        SEARXNG_SECRET=$(openssl rand -hex 32 2>/dev/null || echo "change-me-$(date +%s)")
        cat > "$BASE_DIR/searxng/settings.yml" << SEARXNG_EOF
use_default_settings: true

server:
  secret_key: "$SEARXNG_SECRET"
  bind_address: "0.0.0.0:$SEARXNG_PORT"

search:
  formats:
    - html
    - json
SEARXNG_EOF
        echo "✅ SearXNG settings.yml created at $BASE_DIR/searxng/settings.yml"
    fi

    docker pull searxng/searxng:latest
    docker run -d \
        --name searxng \
        --network host \
        -v "$BASE_DIR/searxng:/etc/searxng:rw" \
        searxng/searxng:latest
    echo "✅ SearXNG starting on http://localhost:$SEARXNG_PORT"
fi

#--- Start OpenWebUI ---
echo "--- Starting OpenWebUI container ---"
docker pull ghcr.io/open-webui/open-webui:main

# Pick the first served NON-ASR model's port as the primary OpenWebUI chat
# endpoint. ASR models are transcription endpoints (not chat), so they must not
# become the primary — they are also skipped in the registration loop below.
OWUI_PRIMARY_PORT=8005
for first_run_idx in "${RUN_SELECTED[@]}"; do
    [ "${MDL_PORT[$first_run_idx]}" = "0" ] && continue
    [ "${MDL_CAT[$first_run_idx]}" = "ASR" ] && continue
    OWUI_PRIMARY_PORT="${MDL_PORT[$first_run_idx]}"
    break
done

OWUI_ENV_ARGS=(
    -e PORT=3000
    -e "OPENAI_API_BASE_URL=http://localhost:${OWUI_PRIMARY_PORT}/v1"
    -e OPENAI_API_KEY=sk-no-key-required
)

if [ -n "$BRAVE_SEARCH_API_KEY" ]; then
    echo "   → Web search: Brave Search API"
    OWUI_ENV_ARGS+=(-e ENABLE_RAG_WEB_SEARCH=true -e WEB_SEARCH_ENGINE=brave -e "BRAVE_SEARCH_API_KEY=$BRAVE_SEARCH_API_KEY")
elif [ "$ENABLE_SEARXNG" = "true" ]; then
    echo "   → Web search: SearXNG (port $SEARXNG_PORT)"
    OWUI_ENV_ARGS+=(-e ENABLE_RAG_WEB_SEARCH=true -e WEB_SEARCH_ENGINE=searxng -e "SEARXNG_QUERY_URL=http://localhost:${SEARXNG_PORT}/search?q=<query>&format=json")
else
    echo "   → Web search: disabled"
fi

docker run -d \
    --name open-webui \
    --network host \
    -v open-webui:/app/backend/data \
    "${OWUI_ENV_ARGS[@]}" \
    ghcr.io/open-webui/open-webui:main

echo "Waiting for OpenWebUI to be ready..."
OWUI_TIMEOUT=300
OWUI_ELAPSED=0
until curl -sf http://localhost:3000/health > /dev/null 2>&1; do
    if [ "$OWUI_ELAPSED" -ge "$OWUI_TIMEOUT" ]; then
        echo ""
        echo "⚠️  OpenWebUI did not become ready after ${OWUI_TIMEOUT}s — check: docker logs open-webui"
        break
    fi
    printf "  [%ds] waiting...\n" "$OWUI_ELAPSED"
    sleep 5
    OWUI_ELAPSED=$((OWUI_ELAPSED + 5))
done

if [ "$OWUI_ELAPSED" -lt "$OWUI_TIMEOUT" ]; then
    echo "✅ OpenWebUI ready at http://localhost:3000"

    # Build URL list dynamically from whatever is actually running
    OWUI_URLS=""
    OWUI_KEYS=""
    OWUI_MANUAL=""
    _owui_add() {
        if [ -z "$OWUI_URLS" ]; then
            OWUI_URLS="\"$1\""; OWUI_KEYS='"sk-no-key-required"'
        else
            OWUI_URLS="$OWUI_URLS,\"$1\""; OWUI_KEYS="$OWUI_KEYS,\"sk-no-key-required\""
        fi
        OWUI_MANUAL="$OWUI_MANUAL\n     $1   ($2)"
    }

    for idx in "${RUN_SELECTED[@]}"; do
        port="${MDL_PORT[$idx]}"
        [ "$port" = "0" ] && continue
        # ASR models expose /v1/audio/transcriptions, not chat — registering them
        # as chat connections would create a broken entry. Wire them into
        # OpenWebUI under Admin → Audio → STT instead.
        if [ "${MDL_CAT[$idx]}" = "ASR" ]; then
            echo "   ℹ️  ${MDL_NAME[$idx]} (port ${port}) is an ASR/transcription endpoint —"
            echo "      skipping chat registration. Add it under Admin → Audio → STT:"
            echo "      OpenAI-compatible URL: http://localhost:${port}/v1"
            continue
        fi
        dir="${MDL_DIR[$idx]}"
        if [ -f "$MODELS_DIR/$dir/config.json" ]; then
            _owui_add "http://localhost:${port}/v1" "${MDL_NAME[$idx]}"
        fi
    done

    if [ -n "$OWUI_ADMIN_EMAIL" ] && [ -n "$OWUI_ADMIN_PASSWORD" ] && [ -n "$OWUI_URLS" ]; then
        echo "--- Auto-registering model connections in OpenWebUI ---"
        OWUI_TOKEN=$(curl -sf -X POST http://localhost:3000/api/v1/auths/signin \
            -H "Content-Type: application/json" \
            -d "{\"email\":\"$OWUI_ADMIN_EMAIL\",\"password\":\"$OWUI_ADMIN_PASSWORD\"}" \
            | jq -r '.token // empty')

        if [ -z "$OWUI_TOKEN" ]; then
            echo "   Sign-in failed — attempting to create admin account..."
            curl -sf -X POST http://localhost:3000/api/v1/auths/signup \
                -H "Content-Type: application/json" \
                -d "{\"name\":\"Admin\",\"email\":\"$OWUI_ADMIN_EMAIL\",\"password\":\"$OWUI_ADMIN_PASSWORD\"}" \
                > /dev/null
            OWUI_TOKEN=$(curl -sf -X POST http://localhost:3000/api/v1/auths/signin \
                -H "Content-Type: application/json" \
                -d "{\"email\":\"$OWUI_ADMIN_EMAIL\",\"password\":\"$OWUI_ADMIN_PASSWORD\"}" \
                | jq -r '.token // empty')
        fi

        if [ -n "$OWUI_TOKEN" ]; then
            curl -sf -X POST http://localhost:3000/api/v1/openai/config/update \
                -H "Authorization: Bearer $OWUI_TOKEN" \
                -H "Content-Type: application/json" \
                -d "{\"ENABLE_OPENAI_API\":true,\"OPENAI_API_BASE_URLS\":[$OWUI_URLS],\"OPENAI_API_KEYS\":[$OWUI_KEYS]}" \
                > /dev/null
            echo "✅ All model connections registered in OpenWebUI"
            printf "   Registered:%b\n" "$OWUI_MANUAL"
        else
            echo "⚠️  OpenWebUI login failed — check OWUI_ADMIN_EMAIL / OWUI_ADMIN_PASSWORD"
        fi
    elif [ -z "$OWUI_URLS" ]; then
        echo "⚠️  No models are running — nothing to register in OpenWebUI."
    else
        echo ""
        echo "   Set OWUI_ADMIN_EMAIL and OWUI_ADMIN_PASSWORD to auto-register connections."
        echo "   Or add them manually: Admin Settings → Connections → + Add Connection"
        printf "%b\n" "$OWUI_MANUAL"
    fi

    echo ""
    echo "  ⏳ Allow 5-10 minutes for vLLM models to finish loading before they appear."
fi

#---------------------------------------------------------------------------------------------------------------
#--- SQLite Structured Memory ---
if [ "$ENABLE_SQLITE_MEMORY" = "true" ]; then
    echo "--- Setting up SQLite structured memory ---"
    _setup_sqlite_memory
fi

#--- Qdrant Vector Memory ---
if [ "$ENABLE_QDRANT" = "true" ]; then
    echo "--- Starting Qdrant vector DB ---"
    _setup_qdrant
fi

#--- Sleep Watchdog ---
# Each model's effective idle timeout is its catalog SLEEP_MIN override, or the
# global IDLE_SLEEP_MINUTES when no override is set. A model with an effective
# timeout of 0 (or empty global) is skipped, so per-model timeouts still apply
# even if the global default is disabled.
if [ "${#RUN_SELECTED[@]}" -gt 0 ]; then
    _build_watch_pairs
    if [ "${#_WATCH_PAIRS[@]}" -gt 0 ]; then
        echo "--- Starting vLLM sleep watchdog ---"
        _start_sleep_watchdog "${_WATCH_PAIRS[@]}"
    else
        echo "--- Sleep watchdog skipped (no served model has an idle-sleep timeout) ---"
    fi
fi

echo ""
echo "--- Disk usage: $BASE_DIR ---"
du -sh "$BASE_DIR" 2>/dev/null
echo ""
echo "--- Per-model breakdown ---"
du -sh "$MODELS_DIR"/*/  2>/dev/null | sort -rh

echo ""
nvidia-smi

echo ""
echo "---- Monitor vLLM Startups ----"
echo "  Run any of the following to tail a model's log:"
for idx in "${RUN_SELECTED[@]}"; do
    port="${MDL_PORT[$idx]}"
    [ "$port" = "0" ] && continue
    echo "    ${MDL_NAME[$idx]} (port ${port}):"
    echo "      tail -f $VLLM_LOGS/vllm-${port}.log"
done
echo ""

# Final snapshot — memory + which models have come up so far.
_show_vllm_status "FINAL"

# Explicit per-model UP / NOT-READY check with test + log commands.
_verify_served_models
echo "  ℹ️  vLLM models take ~5-10 min to finish loading; any 'NOT READY' above are"
echo "     likely still starting. Re-run the health check any time with: $0 --health"
echo ""
