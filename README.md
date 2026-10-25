# Hi, I'm Vineetha 👋

ML engineer (M.S. ECE, USC, Dec 2026, available Jan 2027) working where models meet real constraints: LLM reliability, on-device inference, and healthcare ML. Currently on Alcon's R&D data science team, building GenAI tools on Amazon Bedrock in an FDA-regulated environment.

🔗 [Portfolio](https://vineetha00.github.io) · [LinkedIn](https://www.linkedin.com/in/vineetha-v00) · vineethavc25@gmail.com

## LLM reliability

**[agentaudit](https://github.com/vineetha00/agentaudit)**: "git blame" for multi-agent LLM pipelines. Bisection-based failure attribution (~6 judge calls instead of 20 on long traces), handoff scoring, loop and cost guardrails, LangGraph and CrewAI adapters, and a CI gate. The judge is validated on a 60-trace benchmark with planted failures: Claude Sonnet reaches 86.7% attribution accuracy (Cohen's kappa 0.92). `pip install agentaudit-eval`

**[promptops](https://github.com/vineetha00/promptops)**: Git-native version control for LLM prompts. Semantic diff, Thompson Sampling A/B tests, an MCP server, and a CI regression gate with three triggers (paired t-test, per-case collapse, per-dimension floor), calibrated on 90 run-to-run comparisons. Install from the [v0.3.0 release](https://github.com/vineetha00/promptops/releases/tag/v0.3.0).

**[hearth](https://github.com/vineetha00/hearth)**: a voice-first MCP server for Alexa+, built for the Amazon Developer Hackathon 2026. Amazon Bedrock handles language; deterministic code owns dates, dedupe and ranking, with an offline fallback.

## On-device and efficient inference

**[visionflow](https://github.com/vineetha00/visionflow)**: a quantized vision-language pipeline with hardware-aware backends (CUDA INT4/INT8, Apple MPS, CPU GGUF). Grammar-constrained decoding raised valid structured output from 2% to 79% on 150 DocVQA documents; INT4 runs at 47.7 tok/s in 1.85 GB of VRAM on an L40S, and the ONNX vision encoder runs at 11.90 ms p50 on TensorRT (109.8x over CPU, 1.62x over the CUDA EP), with a harness that caught two silent CPU fallbacks posing as TensorRT. `pip install visionflow-edge` · [eval dataset](https://huggingface.co/datasets/Vineetha00/visionflow)

**[qualcomm-aihub](https://github.com/vineetha00/qualcomm-aihub)**: models taken from PyTorch to the Hexagon NPU through Qualcomm AI Hub. Qwen3-0.6B W4A16 compiled into QNN context binaries for a Snapdragon 8 Elite: 97 tok/s decode at 1K context, 40 tok/s at 4K, 29 ms to first token, 100% of ops on the NPU. INT8 MobileNetV2 on a Galaxy S24: 0.37 to 0.28 ms with 105/105 layers NPU-resident, at a 0.18 pt top-1 cost measured only after verifying FP32 parity with PyTorch.

**[SynapNet-Edge](https://github.com/vineetha00/SynapNet-Edge)**: a hybrid SSM + sparse-attention + episodic-memory model with component-aware quantization. Benchmarked on a real Raspberry Pi 4 (16.6 to 10.2 MB at INT8, within 2 points of FP16), with real INT8 ONNX kernels (1.54x faster single-thread on Apple M3) and a 3.3K-parameter KV-cache eviction policy on par with H2O, SnapKV and PyramidKV. ([architecture repo](https://github.com/vineetha00/SynapNet_Exp))

## Applied ML

- 🧠 [Brain tumor MRI classification](https://github.com/vineetha00/brain-tumor-mri-classification-pytorch-gradcam): fine-tuned ResNet18 with Grad-CAM, 98.8% accuracy on 1,311 held-out images · [live demo](https://huggingface.co/spaces/Vineetha00/Brain-Tumor-Capstone)
- 🛒 [Product recommender](https://github.com/vineetha00/ecommerce-user-product-recommendation-svd): on 3.1M real e-commerce events, found SVD ranked worse than recommending bestsellers, then switched to item-item collaborative filtering for 3.2x better precision@5

## Stack

`Python` `C/C++` `CUDA` `PyTorch` `Hugging Face` `ONNX Runtime` `llama.cpp` `Qualcomm AI Hub` `Amazon Bedrock` `SageMaker` `FastAPI` `Docker` `Kubernetes` `MCP` `LangGraph`
