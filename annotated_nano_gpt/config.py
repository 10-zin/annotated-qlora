from dataclasses import dataclass

@dataclass
class Config:
    name: str = ""
    block_size: int = 4096 
