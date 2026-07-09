from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault("CHROMA_DB_PATH", str((PROJECT_ROOT / "data" / "chroma_db").resolve()))
os.environ.setdefault("RAW_DATA_PATH", str((PROJECT_ROOT / "data" / "raw").resolve()))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings


TOP_K = 8
RERANK_TOP_N = 5
DATASET_PATH = PROJECT_ROOT / "eval" / "test_dataset.json"
RAW_DATA_PATH = Path(settings.raw_data_path)


def normalize_source(source: str | None) -> str:
    if not source:
        return ""
    return Path(str(source).strip().replace("\\", "/")).name.lower()


def ensure_api_key() -> None:
    if not settings.dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请先配置环境变量后再运行评测。")


def build_store() -> None:
    from app.rag.loader import load_documents
    from app.rag.splitter import split_documents
    from app.rag.vector_store import add_documents_to_vector_store, reset_vector_store

    ensure_api_key()
    documents = load_documents(RAW_DATA_PATH)
    if not documents:
        raise RuntimeError(f"{RAW_DATA_PATH} 下没有可用文档，无法重建向量库。")
    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError("文档切分结果为空，无法重建向量库。")
    reset_vector_store()
    add_documents_to_vector_store(chunks)
    print(f"向量库已重建：原始文档={len(documents)}，chunks={len(chunks)}")


def generate_dataset() -> list[dict[str, Any]]:
    samples = load_dataset()
    DATASET_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"测试集已规范化：{DATASET_PATH}，样本数={len(samples)}")
    return samples


def load_dataset() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"测试集不存在：{DATASET_PATH}")
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("测试集格式错误：应为非空列表。")
    for item in samples:
        if "question" not in item or "gold_sources" not in item:
            raise ValueError("每条样本都必须包含 question 和 gold_sources。")
        if not isinstance(item["gold_sources"], list) or not item["gold_sources"]:
            raise ValueError("gold_sources 必须是非空列表。")
    return samples


def collect_retrieved_chunk_sources(docs: list[Any]) -> list[str]:
    """按检索结果顺序返回 source，保留同一文档的重复 chunks。"""
    sources: list[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = normalize_source(metadata.get("source"))
        if source:
            sources.append(source)
    return sources


def collect_retrieved_sources(docs: list[Any]) -> list[str]:
    """返回去重后的 source，仅用于辅助文档级指标。"""
    sources: list[str] = []
    for source in collect_retrieved_chunk_sources(docs):
        if source not in sources:
            sources.append(source)
    return sources


def build_eval_retriever(use_rerank: bool = False):
    if use_rerank:
        from app.rag.reranker import build_rerank_retriever

        return build_rerank_retriever(top_k=TOP_K, rerank_top_n=RERANK_TOP_N)

    from app.rag.retriever import build_hybrid_retriever

    return build_hybrid_retriever(top_k=TOP_K)


def first_relevant_rank(retrieved_sources: list[str], gold_sources: set[str]) -> int | None:
    for index, source in enumerate(retrieved_sources, start=1):
        if source in gold_sources:
            return index
    return None


def summarize_ranks(
    chunk_ranks: list[int | None],
    document_ranks: list[int | None],
    total: int,
    recall_k: int,
) -> dict[str, float]:
    def rate(predicate, ranks: list[int | None]) -> float:
        return sum(1 for rank in ranks if predicate(rank)) / total if total else 0.0

    def mrr(ranks: list[int | None]) -> float:
        return sum(1 / rank for rank in ranks if rank) / total if total else 0.0

    return {
        "chunk_top1": rate(lambda rank: rank == 1, chunk_ranks),
        "chunk_top3": rate(lambda rank: rank is not None and rank <= 3, chunk_ranks),
        "chunk_recall_at_5": rate(lambda rank: rank is not None and rank <= recall_k, chunk_ranks),
        "chunk_mrr": mrr(chunk_ranks),
        "document_top1": rate(lambda rank: rank == 1, document_ranks),
        "document_top3": rate(lambda rank: rank is not None and rank <= 3, document_ranks),
        "document_recall_at_5": rate(lambda rank: rank is not None and rank <= recall_k, document_ranks),
        "document_mrr": mrr(document_ranks),
    }


def evaluate_dataset(
    samples: list[dict[str, Any]],
    *,
    use_rerank: bool = False,
    quiet: bool = False,
) -> dict[str, float]:
    ensure_api_key()
    retriever = build_eval_retriever(use_rerank=use_rerank)
    chunk_ranks: list[int | None] = []
    document_ranks: list[int | None] = []
    misses: list[dict[str, Any]] = []

    for index, item in enumerate(samples, start=1):
        question = item["question"]
        gold_sources = {normalize_source(source) for source in item["gold_sources"]}
        docs = retriever.invoke(question) or []
        retrieved_chunks = collect_retrieved_chunk_sources(docs)
        retrieved_sources = collect_retrieved_sources(docs)
        chunk_rank = first_relevant_rank(retrieved_chunks, gold_sources)
        document_rank = first_relevant_rank(retrieved_sources, gold_sources)
        chunk_ranks.append(chunk_rank)
        document_ranks.append(document_rank)

        if chunk_rank is None:
            misses.append(
                {
                    "id": item.get("id", str(index)),
                    "question": question,
                    "gold_sources": sorted(gold_sources),
                    "retrieved_sources": retrieved_sources,
                }
            )

        if not quiet:
            print("-" * 80)
            print(f"样本: {index}/{len(samples)} {item.get('id', '')}")
            print(f"问题: {question}")
            print(f"标准文档: {sorted(gold_sources)}")
            print(f"召回 chunks: {retrieved_chunks}")
            print(f"chunk 首个相关排名: {chunk_rank or '未命中'}")
            print(f"文档首个相关排名: {document_rank or '未命中'}")

    recall_k = RERANK_TOP_N if use_rerank else TOP_K
    summary = summarize_ranks(chunk_ranks, document_ranks, len(samples), recall_k)
    print("=" * 80)
    print(f"评测模式: {'hybrid + reranker' if use_rerank else 'hybrid retrieval'}")
    print(f"样本总数: {len(samples)}")
    print(f"严格 chunk Top1: {summary['chunk_top1']:.4f}")
    print(f"严格 chunk Top3: {summary['chunk_top3']:.4f}")
    print(f"严格 chunk Recall@{recall_k}: {summary['chunk_recall_at_5']:.4f}")
    print(f"严格 chunk MRR: {summary['chunk_mrr']:.4f}")
    print(f"辅助文档 Top1: {summary['document_top1']:.4f}")
    print(f"辅助文档 Top3: {summary['document_top3']:.4f}")
    print(f"辅助文档 Recall@{recall_k}: {summary['document_recall_at_5']:.4f}")
    print(f"辅助文档 MRR: {summary['document_mrr']:.4f}")
    print("说明：主指标按 chunk 排名计算，文档级指标只用于观察 source 聚合后的结果。")

    if misses:
        print("未命中样本汇总:")
        for item in misses:
            print(f"- {item['id']}: gold={item['gold_sources']} retrieved={item['retrieved_sources']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测工具")
    parser.add_argument("action", nargs="?", default="all", choices=["build_store", "generate", "evaluate", "all"])
    parser.add_argument("--rerank", action="store_true", help="启用本地 CrossEncoder 重排，速度较慢")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条样本")
    parser.add_argument("--quiet", action="store_true", help="不打印逐题明细")
    args = parser.parse_args()

    if args.action == "build_store":
        build_store()
        return
    if args.action == "generate":
        generate_dataset()
        return

    samples = load_dataset()
    if args.limit > 0:
        samples = samples[: args.limit]
    evaluate_dataset(samples, use_rerank=args.rerank, quiet=args.quiet)


if __name__ == "__main__":
    main()
