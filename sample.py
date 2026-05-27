#!/usr/bin/env python3
"""
Sample n outputs from a local Ollama model, embed them, run PCA + Archetypal Analysis,
and produce a 3D plot. Edit DEFAULT_PROMPT below and just run:  python3 sample.py
"""

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.optimize import nnls
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

DEFAULT_PROMPT = "Tell me a joke about programming"
DEFAULT_N = 100
DEFAULT_ARCHETYPES = 4
DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def query_ollama_api(payload: dict, host: str) -> str:
    api_url = f"{host}/api/generate"
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Query failed with status {response.status_code}: {response.text}")
    result = response.json()
    return result.get("response", "")


def query_ollama_embedding(text: str, model: str, host: str) -> list[float]:
    api_url = f"{host}/api/embeddings"
    payload = {"model": model, "prompt": text}
    response = requests.post(api_url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Embedding query failed with status {response.status_code}: {response.text}")
    result = response.json()
    return result.get("embedding", [])


def run_archetypal_analysis(X: np.ndarray, k: int, max_iter: int = 200, tol: float = 1e-6):
    """
    Archetypal analysis via alternating NNLS.
    X: (n, m) data matrix.  Returns: archetypes Z (k, m), alpha (n, k), beta (k, n), error.
    """
    n, m = X.shape

    rng = np.random.RandomState(0)
    indices = rng.choice(n, k, replace=False)
    Z = X[indices].copy()

    lam = float(np.linalg.norm(X)) * 100

    for iteration in range(max_iter):
        Z_old = Z.copy()

        Z_aug = np.vstack([Z.T, lam * np.ones((1, k))])

        alpha = np.zeros((n, k))
        for i in range(n):
            x_aug = np.r_[X[i], lam]
            coeffs, _ = nnls(Z_aug, x_aug)
            s = coeffs.sum()
            alpha[i] = coeffs / s if s > 0 else coeffs

        X_aug = np.vstack([X.T, lam * np.ones((1, n))])

        beta = np.zeros((k, n))
        for j in range(k):
            z_aug = np.r_[Z[j], lam]
            coeffs, _ = nnls(X_aug, z_aug)
            s = coeffs.sum()
            beta[j] = coeffs / s if s > 0 else coeffs

        Z = beta @ X

        change = np.linalg.norm(Z - Z_old) / max(1e-12, np.linalg.norm(Z_old))
        if change < tol:
            break

    recon = alpha @ Z
    error = float(np.mean(np.sum((X - recon) ** 2, axis=1)))

    return Z, alpha, beta, error


def main():
    parser = argparse.ArgumentParser(description="Sample, embed, PCA, and Archetypal Analysis of LLM outputs.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help=f"Input prompt (default: from DEFAULT_PROMPT).")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"Number of samples (default: {DEFAULT_N}).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8). Must be > 0.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens per sample (default: 100).")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST, help=f"Ollama URL (default: {DEFAULT_OLLAMA_HOST}).")
    parser.add_argument("--embed-model", type=str, default="nomic-embed-text", help="Embedding model (default: nomic-embed-text).")
    parser.add_argument("--archetypes", type=int, default=DEFAULT_ARCHETYPES, help=f"Number of archetypes (default: {DEFAULT_ARCHETYPES}, 0 to skip).")
    parser.add_argument("--output", type=str, default="pca_3d.png", help="Output plot filename (default: pca_3d.png).")
    args = parser.parse_args()

    if args.temperature <= 0:
        print("Error: temperature must be greater than 0.", file=sys.stderr)
        sys.exit(1)

    host = args.ollama_host.rstrip("/")
    print(f"Prompt: '{args.prompt}'")
    print(f"Samples: {args.n}  |  Model: {args.model}  |  Temperature: {args.temperature}")
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

    # --- Embeddings ---
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

    if not embeddings:
        print("No embeddings generated. Exiting.", file=sys.stderr)
        sys.exit(1)

    X = np.array(embeddings)
    n_samples, n_features = X.shape

    # --- PCA ---
    n_components = min(n_samples, n_features, 5)
    print(f"Running PCA ({n_samples} samples, {n_features} dims -> {n_components} components)...")
    print("-" * 50)

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

    # --- Convex Hull of 3D PCA projections ---
    points_3d = projected[:, :3]
    hull = ConvexHull(points_3d)
    hull_vertex_mask = np.zeros(n_samples, dtype=bool)
    hull_vertex_mask[hull.vertices] = True

    print()
    print(f"Convex hull (3D PCA space):")
    print(f"  Volume: {hull.volume:.6f}")
    print(f"  Surface area: {hull.area:.6f}")
    print(f"  Vertices: {len(hull.vertices)} of {n_samples} samples")
    hull_indices = sorted(v + 1 for v in hull.vertices)
    print(f"  Hull sample indices: {hull_indices}")

    # --- Archetypal Analysis ---
    arch_projected = None
    arch_alpha = None
    arch_beta = None
    arch_error = None
    arch_labels = None

    if args.archetypes > 0 and args.archetypes <= n_samples:
        k = args.archetypes
        print(f"\nRunning Archetypal Analysis (k={k})...")
        print("-" * 50)

        Z_arch, alpha, beta, error = run_archetypal_analysis(X, k)

        arch_projected = pca.transform(Z_arch)
        arch_alpha = alpha
        arch_beta = beta
        arch_error = error

        print(f"Reconstruction error (per-dim MSE): {error:.6f}")
        print()

        # Find closest sample to each archetype
        print("Archetype summaries:")
        for j in range(k):
            dists = np.sum((X - Z_arch[j]) ** 2, axis=1)
            closest = int(np.argmin(dists))
            preview = generated_texts[closest][:80].replace("\n", " ")
            print(f"  Archetype {j+1}: closest sample {closest+1} | \"{preview}...\"")

        # Dominant archetype per sample
        dominant = np.argmax(alpha, axis=1)
        print()
        print("Dominant archetype per sample:")
        for j in range(n_samples):
            print(f"  Sample {j+1}: Archetype {dominant[j]+1} (weight {alpha[j, dominant[j]]:.3f})")
    elif args.archetypes > n_samples:
        print(f"\nWarning: archetypes ({args.archetypes}) > samples ({n_samples}), skipping.", file=sys.stderr)

    # --- 3D Plot ---
    print(f"\nGenerating 3D plot...")
    print("-" * 50)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    xs = projected[:, 0]
    ys = projected[:, 1]
    zs = projected[:, 2] if projected.shape[1] >= 3 else np.zeros_like(xs)

    interior = ~hull_vertex_mask
    ax.scatter(xs[interior], ys[interior], zs[interior], c="steelblue", s=60, label="Interior", alpha=0.5)
    ax.scatter(xs[hull_vertex_mask], ys[hull_vertex_mask], zs[hull_vertex_mask],
               c="orange", marker="o", s=120, edgecolors="black", linewidths=0.8, label=f"Hull vertices ({len(hull.vertices)})")
    for j in range(len(xs)):
        ax.text(xs[j], ys[j], zs[j], f"  {j+1}", size=9, color="black" if hull_vertex_mask[j] else "gray")

    if arch_projected is not None:
        axs = arch_projected[:, 0]
        ays = arch_projected[:, 1]
        azs = arch_projected[:, 2] if arch_projected.shape[1] >= 3 else np.zeros_like(axs)
        ax.scatter(axs, ays, azs, c="red", marker="*", s=400, label="Archetypes", edgecolors="black", linewidths=0.5)
        for j in range(len(axs)):
            ax.text(axs[j], ays[j], azs[j], f"  A{j+1}", size=12, weight="bold", color="red")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"LLM Response Embeddings  |  Hull vol={hull.volume:.4f}")
    ax.legend()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to {args.output}")

    # --- Convex Hull Plot ---
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection="3d")

    ax2.plot_trisurf(xs, ys, zs, triangles=hull.simplices,
                     color="orange", alpha=0.2, edgecolor="orange", linewidth=0.5)
    ax2.scatter(xs[hull_vertex_mask], ys[hull_vertex_mask], zs[hull_vertex_mask],
                c="orange", marker="o", s=120, edgecolors="black", linewidths=0.8, label=f"Hull vertices ({len(hull.vertices)})")
    for j in hull.vertices:
        ax2.text(xs[j], ys[j], zs[j], f"  {j+1}", size=10, weight="bold")

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title(f"Convex Hull  |  Volume = {hull.volume:.4f}")
    ax2.legend()
    fig2.savefig("convex_hull.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Hull plot saved to convex_hull.png")


if __name__ == "__main__":
    main()
