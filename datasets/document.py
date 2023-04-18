from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Document is a class that contains a document's title, author, content, and post time."""
    title: str
    author: str
    content: str
    post_time: datetime
