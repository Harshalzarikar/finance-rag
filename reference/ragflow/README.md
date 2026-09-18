# Reference only — upstream RAGFlow deployment assets.
#
# `docker-compose.ragflow.yml` and `env.ragflow.example` are the upstream
# InfiniFlow RAGFlow v0.27.2 stack (Elasticsearch/Infinity, MySQL, MinIO, Redis,
# NATS, ClickHouse, ragflow-server). They are kept here because this project
# borrows RAGFlow's *design* — DeepDoc layout recognition, section-boundary
# chunking, hybrid recall, fused reranking — but does not run RAGFlow itself.
#
# They are NOT part of this deployment and will not start as-is: the compose file
# includes ./docker-compose-base.yml and mounts service_conf.yaml.template and
# entrypoint.sh, none of which were copied into this repository.
#
# Design ideas adopted from RAGFlow:
#   - src/ingestion/deepdoc_loader.py   layout-aware parsing (DLR/TSR) via the
#                                       same family of ONNX models
#   - src/ingestion/chunker.py          parent/child chunking on section boundaries
#   - src/retrieval/hybrid_search.py    "multiple recall": keyword + dense, fused
#   - src/llm/reranker.py               cross-encoder reranking of the fused set
