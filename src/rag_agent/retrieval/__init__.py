"""Hybrid retrieval: vector search (ChromaDB) + BM25 keyword search, fused
by configurable weights, then reranked by an ONNX cross-encoder.
See hybrid.py for the orchestrator.
"""
