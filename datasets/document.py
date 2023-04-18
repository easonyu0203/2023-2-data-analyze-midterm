from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple


@dataclass
class Document:
    """Document is a class that contains a document's title, author, content, and post time."""
    title: str
    author: str
    content: str
    post_time: datetime
    keywords: List[Tuple[str, float]] | None = None
