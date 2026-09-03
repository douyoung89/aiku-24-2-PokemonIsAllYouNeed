# Pokémon Is All You Need

A multimodal video-retrieval pipeline that extracts Pokémon animation highlights from a natural-language query by combining visual detection with semantic subtitle matching.

> This project was completed in Fall 2024 as part of [AIKU](https://github.com/AIKU-Official), an artificial intelligence student organization at Korea University.

![Project introduction](./assets/Introduction.png)

## Overview

Finding every scene that features a specific Pokémon or action requires manually searching through many episodes. This project automates that process.

Given a Korean-language query such as **"피카츄가 싸운다" ("Pikachu is fighting")**, the system:

1. identifies the Pokémon mentioned in the query;
2. retrieves frames containing that Pokémon through visual similarity;
3. retrieves semantically relevant subtitle segments; and
4. combines both signals to extract candidate highlight clips.

## My Contributions

**Douyoung Kwon ([douyoung89](https://github.com/douyoung89))**

- Contributed to the Pokémon detection pipeline used to identify candidate character regions.
- Developed subtitle-based semantic retrieval using sentence embeddings and cosine similarity.
- Participated in the design and implementation of the multimodal highlight-extraction workflow.

## Method

![Multimodal retrieval pipeline](./assets/Pipeline.png)

The pipeline consists of two complementary retrieval tracks.

### 1. Visual Retrieval

1. Run object detection over animation frames to obtain candidate Pokémon bounding boxes.
2. Encode each detected crop with the image encoder from [OpenCLIP](https://github.com/mlfoundations/open_clip).
3. Parse the Pokémon name from the user query and encode reference images of that Pokémon with the same encoder.
4. Calculate cosine similarity between the reference and frame embeddings.
5. Retain frames whose similarity exceeds a configurable threshold.

The current implementation uses the OpenCLIP **ViT-g/14** model pretrained on **LAION-2B**.

### 2. Subtitle-Based Semantic Retrieval

1. Encode the user query with the Korean **KoE5** Sentence Transformer.
2. Encode timestamped subtitle segments from the animation.
3. Calculate cosine similarity between the query and subtitle embeddings.
4. Select timestamps whose semantic similarity exceeds a configurable threshold.

Subtitle embeddings are cached to avoid recomputing them across queries.

### 3. Multimodal Fusion and Clip Extraction

The system converts visually matched frames into candidate time intervals and cross-checks them against subtitle-derived timestamps. Intervals supported by both modalities are padded and extracted from the source video with OpenCV.

This fusion helps narrow visual matches to scenes that also align with the action or context described by the user.

## Example Results

**Input:** "피카츄가 싸운다" ("Pikachu is fighting")

<p>
  <img src="./assets/result1.gif" alt="Retrieved Pikachu highlight 1" width="200">
  <img src="./assets/result2.gif" alt="Retrieved Pikachu highlight 2" width="200">
  <img src="./assets/result3.gif" alt="Retrieved Pikachu highlight 3" width="200">
</p>

## Repository Contents

| File | Description |
| --- | --- |
| `pipeline.ipynb` | End-to-end retrieval, fusion, and video-extraction workflow |
| `extract_embedding.py` | OpenCLIP image-embedding utilities |
| `transform_subtitles.py` | Timestamped subtitle preprocessing |
| `assets/` | Architecture figures and qualitative result examples |

## Environment

The implementation is based on Python and the following libraries:

- [PyTorch](https://pytorch.org/)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Sentence Transformers](https://www.sbert.net/)
- scikit-learn
- OpenCV
- Pillow
- NumPy
- Matplotlib

A CUDA-capable GPU is recommended for image-embedding extraction.

## Usage

1. Prepare the animation videos, detected Pokémon crops, timestamped subtitles, Pokémon name mapping, and precomputed frame embeddings.
2. Update the local data paths in `pipeline.ipynb` and `transform_subtitles.py`.
3. Set `user_input` in `pipeline.ipynb` to a Korean-language description.
4. Run the notebook cells in sequence to retrieve matching timestamps and export highlight clips.

### Data Availability and Reproducibility

The animation episodes and derived data are not distributed in this repository because of data and copyright constraints. Consequently, the notebook is not executable out of the box. The repository documents the research prototype and provides the core retrieval and processing code.

## Team

| Member | Role |
| --- | --- |
| [Changyeop Lee](https://github.com/PROLCY) | Team lead; image-embedding pipeline |
| [Douyoung Kwon](https://github.com/douyoung89) | Pokémon detection; subtitle-based retrieval |
| [Minjun Kim](https://github.com/ddomjun) | Pokémon detection; subtitle-based retrieval |
| [Moo-geun Park](https://github.com/MooGeunPark) | Dataset collection; subtitle-based retrieval |
