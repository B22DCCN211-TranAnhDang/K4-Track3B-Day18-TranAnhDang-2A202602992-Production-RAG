"""
Interactive Chatbot CLI for Lab 18 Production RAG.
Chạy: python chat.py
"""

import sys, os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.pipeline import build_pipeline, run_query


def main():
    print("=" * 60)
    print("🤖 CHATBOT HỎI ĐÁP NỘI BỘ (PRODUCTION RAG PIPELINE)")
    print("=" * 60)
    print("Đang khởi tạo index và nạp các mô hình AI...")
    
    search, reranker = build_pipeline()
    
    print("\n✅ Khởi tạo xong! Nhập câu hỏi của bạn (Gõ 'exit' hoặc 'quit' để thoát).\n")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n💬 Bạn: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break

            print("🔍 Đang tra cứu quy định và xếp hạng ngữ cảnh...")
            answer, contexts = run_query(query, search, reranker)
            
            print("\n🤖 Chatbot:")
            print(answer)
            print("\n📚 Nguồn ngữ cảnh trích xuất (Top 3):")
            for i, ctx in enumerate(contexts, 1):
                clean_ctx = ctx.replace("\n", " ")[:150]
                print(f"  [{i}] {clean_ctx}...")
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
