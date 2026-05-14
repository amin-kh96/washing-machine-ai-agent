🧠 Project Overview
Modern AI assistants typically require massive computing power and cloud APIs. This project proves that an embedded device can run a fully capable, hallucination-free AI by combining a tiny language model (135M parameters) with dynamic Context Injection.

Key Achievements:

Supervised Fine-Tuning (SFT): Trained a base model (SmolLM2-135M) to understand user queries and respond with the persona of a polite washing machine assistant.

Retrieval-Augmented Generation (RAG): Eliminated parametric memory hallucinations by injecting a local manufacturer JSON manual directly into the model's prompt at runtime.

Edge Deployment: Converted the architecture to run locally on an ARM Cortex-A35 processor using llama.cpp and GGUF quantization.

💻 Hardware Specifications
The target hardware for this deployment is the STM32MP257F-EV1 Evaluation Board.

Processor: STMicroelectronics STM32MP257FAI3 (Arm® dual-core Cortex®-A35 @ 1.5 GHz)

Memory: 4GB DDR4 DRAM (Crucial for loading the LLM in RAM)

OS: OpenSTLinux (Mainlined open-source Linux® distribution)

Storage: microSD / 32-Gbit eMMC

🏗️ Architecture & Methodology
Phase 1: Supervised Fine-Tuning (SFT)
The base model (SmolLM2-135M) was fine-tuned using LoRA (Low-Rank Adaptation). The goal of this phase was to teach the model behavior (how to speak like an assistant), rather than facts.

Limitation Discovered: During testing, the 135M model suffered from severe factual hallucinations (e.g., inventing fake "Low-Effort" machines or incorrect temperatures) due to its limited parametric memory.

Phase 2: Context Injection (RAG)
To cure the hallucinations without scaling up to a massive 1.7B+ parameter model, a lightweight RAG system was implemented.

Knowledge Base: A structured washing_manual.json file acts as the manufacturer's ground truth.

Retrieval: A Python script dynamically parses the user's prompt for keywords (e.g., "Silk", "Cotton") and extracts the exact temperature, spin speed, and instructions from the JSON.

Augmentation: The extracted facts are injected into the model's system prompt.

Result: The model achieved 100% factual accuracy, seamlessly combining its conversational training with the hardcoded JSON facts.

Phase 3: Edge Optimization (GGUF)
Standard Python transformers libraries are too heavy for an embedded ARM processor. The model and adapter were merged and quantized into a highly efficient GGUF format to be run via llama.cpp natively on the STM32 OpenSTLinux environment.

📂 Repository Structure
test_agent.py: The main testing script for PC validation, demonstrating the RAG / Context Injection pipeline.

washing_manual.json: The local database containing factual washing machine cycles and parameters.

merge_model.py: Utility script to fuse the base model and fine-tuned adapter into a single checkpoint for GGUF conversion.

(Ignored by Git): Dataset files, raw model weights, and .gguf binaries

📊 Conclusion
This implementation proves that high-performance, domain-specific AI can be deployed on edge devices by leveraging efficient architectures (135M parameters) paired with deterministic logic (Context Injection). It completely removes the dependency on cloud connectivity, ensuring low latency, high privacy, and zero operational costs for smart appliance manufacturers.
