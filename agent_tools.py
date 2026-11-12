import os
import json
from pathlib import Path


class CardLookupTool:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
        self.cards = {
            "dragon": self._load_cards("dragon_cards.json"),
            "cave": self._load_cards("cave_cards.json"),
        }

    def _load_cards(self, filename: str) -> dict[int, dict]:
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as file:
            card_records = json.load(file)
        return {
            card["number"]: card
            for card in card_records
            if isinstance(card, dict) and isinstance(card.get("number"), int)
        }

    def lookup_card(self, card_type: str, card_id: int | str) -> str:
        """Return the JSON details for a dragon or cave card by its card number."""
        normalized_type = str(card_type).strip().lower()
        if normalized_type.endswith("_card"):
            normalized_type = normalized_type[:-5]
        if normalized_type not in self.cards:
            return "Error: card_type must be 'dragon' or 'cave'."

        if isinstance(card_id, bool):
            return "Error: card_id must be an integer."
        try:
            normalized_id = int(card_id)
        except (TypeError, ValueError):
            return "Error: card_id must be an integer."

        card = self.cards[normalized_type].get(normalized_id)
        if card is None:
            return f"No {normalized_type} card found with id {normalized_id}."

        return f"{normalized_type.title()} card #{normalized_id}:\n{json.dumps(card, indent=2)}"


class RulebookSearchTool:
    def __init__(self, persist_dir: str = "./agents/wyrmspan_vector_store"):
        "Initializes the search tool by loading the FAISS vector store from the specified directory."
        from langchain_community.embeddings import HuggingFaceEmbeddings

        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self._load_index()

    def _load_index(self):
        from langchain_community.vectorstores import FAISS

        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(f"The specified persist directory does not exist: {self.persist_dir}")
        
        print(f"> Search Tool - Loading FAISS vector store from: {self.persist_dir}")
        self.vector_store = FAISS.load_local(
            self.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def search_rules(self, query: str, top_k: int = 5) -> str:
        """
        Searches the FAISS vector store for the most relevant rulebook entries based on the query.
        Returns formatted text complete with source filenames and section headers.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
        """
        if not self.vector_store:
            self._load_index()

        # execute similarity search
        results = self.vector_store.similarity_search(query, k=top_k)

        if not results:
            return "No relevant rulebook entries found for this query."

        formatted_output = []
        for i, doc in enumerate(results, start=1):
            source_file = doc.metadata.get("source_file", "unknown_source.md")
            header_1 = doc.metadata.get("Header_1", "")
            header_2 = doc.metadata.get("Header_2", "")
            header_3 = doc.metadata.get("Header_3", "")

            # build a clear hierarchical breadcrumb trail
            hierarchy = [h for h in [header_1, header_2, header_3] if h]
            breadcrumb = " > ".join(hierarchy) if hierarchy else "No headers available"

            location_tag = f"[{source_file}" + (f" | {breadcrumb}]" if breadcrumb else "]")

            # format the chunk cleanly for the LLM's context window
            formatted_output.append(
                f"--- Reference {i} {location_tag} ---\n{doc.page_content.strip()}\n"
            )

        return "\n\n".join(formatted_output)

if __name__ == "__main__":
    # Example usage
    search_tool = RulebookSearchTool()
    query = "Mechanics of competing with the automa for guild spaces and benefits"
    results = search_tool.search_rules(query, top_k=3)
    print(results)