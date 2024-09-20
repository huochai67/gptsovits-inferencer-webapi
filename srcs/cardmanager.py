from pathlib import Path

from card import Card
from result import Result, Ok, Err

from fs import getdatadir
from model.manager import ModelManager

modelManager = ModelManager(f"{getdatadir()}/models")

class CardManagerImpl:
    cards: dict = {}

    def __init__(self, dirpath: str):
        for entry in Path(dirpath).iterdir():
            if entry.is_file():
                # try:
                card: Card = Card.from_file(entry, modelManager)
                self.cards[card.name] = card
                # except

    def try_get(self, cardname: str) -> Result[Card, str]:
        if cardname in self.cards:
            return Ok(self.cards[cardname])
        return Err("key not found")

    def get_all(self) -> dict:
        return self.cards


CardManager = CardManagerImpl(f"{getdatadir()}/cards")
