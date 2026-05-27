#!/usr/bin/env python3
"""
Sample n outputs from a local Ollama model.
Usage:
    python sample.py --prompt "Your prompt here" --n 5 [--model llama3.2]
"""

import argparse
import sys
import time

import numpy as np
import requests
from sklearn.decomposition import PCA

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
    parser.add_argument("--pca", action="store_true", help="Run PCA on collected embeddings and show 2D projections.")
    parser.add_argument("--pca-components", type=int, default=5, help="Number of PCA components (default: 5). Clamped to min(samples, features).")
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

    do_pca = args.pca
    if args.embed or do_pca:
        embed_model = args.embed_model
        print(f"\nGenerating embeddings using '{embed_model}'...")
        print("-" * 50)
        embeddings = []
        for i, text in enumerate(generated_texts):
            try:
                start_time = time.time()
                embedding = query_ollama_embedding(text, embed_model, host)
                elapsed = time.time() - start_time
                embeddings.append(embedding)
                dim = len(embedding)
                preview = ", ".join(f"{v:.4f}" for v in embedding[:5])
                print(f"Embedding {i+1}/{len(generated_texts)} (took {elapsed:.2f}s):")
                print(f"  Dimension: {dim}")
                print(f"  First 5 values: [{preview}, ...]")
                print("-" * 50)
            except Exception as e:
                print(f"Error generating embedding {i+1}: {e}", file=sys.stderr)

        if do_pca and embeddings:
            n_samples = len(embeddings)
            n_features = len(embeddings[0])
            n_components = min(n_samples, n_features, args.pca_components)

            print(f"\nRunning PCA ({n_samples} samples, {n_features} dims -> {n_components} components)...")
            print("-" * 50)

            X = np.array(embeddings)
            pca = PCA(n_components=n_components)
            projected = pca.fit_transform(X)

            print("Explained variance ratio per component:")
            for j, ratio in enumerate(pca.explained_variance_ratio_):
                print(f"  PC{j+1}: {ratio:.4f} ({ratio*100:.2f}%)")
            print(f"  Cumulative: {np.sum(pca.explained_variance_ratio_):.4f} ({np.sum(pca.explained_variance_ratio_)*100:.2f}%)")
            print()

            print("2D projections (PC1 vs PC2):")
            for j, (text, proj) in enumerate(zip(generated_texts, projected)):
                preview = text[:60].replace("\n", " ")
                print(f"  Sample {j+1}: PC1={proj[0]:+.4f}, PC2={proj[1]:+.4f}  |  \"{preview}...\"")


if __name__ == "__main__":
    main()
