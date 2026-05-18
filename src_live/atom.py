from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


ATOMIC_NUMBER = {"H": 1, "C": 6, "N":7, "O":8, "B":5}
ELEMENT_SYMBOL = {v:k for k,v in ATOMIC_NUMBER.items()} # dict comprehension, erstellt ein neues dict, indem die keys und values vertauscht werden

@dataclass     # iinit funktion wird damit schon definiert
class Atom:
    symbol: str
    coord: Sequence[float]

    def __post_init__(self) -> None:   #liefert nichts zurück
        self.symbol = self.symbol.strip().capitalize()
        self.coord = np.asarray(self.coord, dtype = float)

    def __str__(self):   #helferfuntionenn dunder-Funktion __name__
        return f"{self.symbol} {self.coord[0]:.6f} {self.coord[1]:.6f} {self.coord[2]:.6f}"
     
    @property
    def atomic_number(self) -> int:
        return ATOMIC_NUMBER[self.symbol]
    
    @property
    def n_electrons(self) -> int:
        return self.atomic_number
    


    @classmethod
    def from_string(cls, s: str) -> Atom:
        parts = s.split()   # spaltet string in liste von substrings
        if len(parts) != 4:
            raise ValueError(f"Expected 'Symbol X Y Z', got {s}")
        symbol = parts[0]   # 0. element der liste
        try:
            coord = np.asarray(parts[1:], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates in atom string {s}") from exc
        return cls(symbol, coord)

    def get_distance(self, other: Atom) -> float:
        if not isinstance(other, Atom): 
            raise ValueError(f"Distance can only be calculated between two Atom intstances")
        return float(np.linalg.norm(self.coord - other.coord))
