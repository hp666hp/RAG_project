import unittest

from langchain_core.documents import Document

from eval.rag_evaluate import collect_retrieved_chunk_sources, summarize_ranks


class RagEvaluateTests(unittest.TestCase):
    def test_chunk_sources_keep_duplicate_chunks_and_original_order(self):
        docs = [
            Document(page_content="a", metadata={"source": "data/raw/01_policy.md"}),
            Document(page_content="b", metadata={"source": "C:/repo/data/raw/01_policy.md"}),
            Document(page_content="c", metadata={"source": "data/raw/02_other.md"}),
        ]

        self.assertEqual(
            collect_retrieved_chunk_sources(docs),
            ["01_policy.md", "01_policy.md", "02_other.md"],
        )

    def test_summary_distinguishes_chunk_and_document_ranks(self):
        summary = summarize_ranks(
            [1, 4, None],
            [1, 1, None],
            total=3,
            recall_k=5,
        )

        self.assertEqual(summary["chunk_top1"], 1 / 3)
        self.assertEqual(summary["document_top1"], 2 / 3)
        self.assertEqual(summary["chunk_recall_at_5"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
