#!/usr/bin/env python3
"""
Sample n outputs from a local Ollama model.
Usage:
    python sample.py --prompt "Your prompt here" --n 5 [--model llama3.2]
"""

import argparse
import sys
import time

import requests

DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def query_ollama_api(payload: dict, host: str) -> str:
    """Query the Ollama API and return generated text."""
    api_url = f"{host}/api/generate"
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Query failed with status {response.status_code}: {response.text}")
    result = response.json()
    return result.get("response", "")


def query_ollama_embedding(text: str, model: str, host: str) -> list[float]:
    """Query the Ollama embedding API and return the embedding vector."""
    api_url = f"{host}/api/embeddings"
    payload = {"model": model, "prompt": text}
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Embedding query failed with status {response.status_code}: {response.text}")
    result = response.json()
    return result.get("embedding", [])


def main():
    parser = argparse.ArgumentParser(description="Sample n outputs from a local Ollama model with nonzero temperature.")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt.")
    parser.add_argument("--n", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8). Must be > 0.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate (default: 100).")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST, help=f"Ollama server URL (default: {DEFAULT_OLLAMA_HOST}).")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings for each sampled response.")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text", help="Model to use for embeddings (default: nomic-embed-text).")
    args = parser.parse_args()

    if args.temperature <= 0:
        print("Error: temperature must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    host = args.ollama_host.rstrip("/")
    print(f"Generating {args.n} samples for prompt: '{args.prompt}'")
    print(f"Model: {args.model}, temperature: {args.temperature}")
    print(f"Ollama host: {host}")
    print("-" * 50)

    generated_texts = []
    for i in range(args.n):
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "temperature": args.temperature,
            "num_predict": args.max_new_tokens,
            "stream": False,
        }
        try:
            start_time = time.time()
            generated_text = query_ollama_api(payload, host)
            elapsed = time.time() - start_time
            generated_texts.append(generated_text)
            print(f"Sample {i+1}/{args.n} (took {elapsed:.2f}s):")
            print(generated_text)
            print("-" * 50)
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}", file=sys.stderr)

    if args.embed:
        embed_model = args.embed_model
        print(f"\nGenerating embeddings using '{embed_model}'...")
        print("-" * 50)
        for i, text in enumerate(generated_texts):
            try:
                start_time = time.time()
                embedding = query_ollama_embedding(text, embed_model, host)
                elapsed = time.time() - start_time
                dim = len(embedding)
                preview = ", ".join(f"{v:.4f}" for v in embedding[:5])
                print(f"Embedding {i+1}/{len(generated_texts)} (took {elapsed:.2f}s):")
                print(f"  Dimension: {dim}")
                print(f"  First 5 values: [{preview}, ...]")
                print("-" * 50)
            except Exception as e:
                print(f"Error generating embedding {i+1}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
