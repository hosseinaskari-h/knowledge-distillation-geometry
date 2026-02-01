"""
Prompt bank for initializing conversations
"""
import random
from typing import List


class PromptBank:
    """Manages conversation starting prompts"""

    def __init__(self, prompts: List[str] = None):
        if prompts is None:
            prompts = self._default_prompts()
        self.prompts = prompts

    def sample(self) -> str:
        """Sample random prompt"""
        return random.choice(self.prompts)

    @staticmethod
    def _default_prompts() -> List[str]:
        """Default prompt set - famous opening lines"""
        return [
            "The sky above the port was the color of television, tuned to a dead channel.",
            "In a hole in the ground there lived a hobbit.",
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "It was the best of times, it was the worst of times.",
            "Call me Ishmael.",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "Happy families are all alike; every unhappy family is unhappy in its own way.",
            "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer.",
            "There was a boy called Eustace Clarence Scrubb, and he almost deserved it.",
            "The drought had lasted now for ten million years, and the reign of the terrible lizards had long since ended.",
            "Far out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the Galaxy lies a small unregarded yellow sun.",
            "In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
            "The story so far: In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
            "A screaming comes across the sky.",
            "Lolita, light of my life, fire of my loins.",
            "riverrun, past Eve and Adam's, from swerve of shore to bend of bay, brings us by a commodius vicus of recirculation back to Howth Castle and Environs.",
            "Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of lather on which a mirror and a razor lay crossed.",
            "Someone must have been telling lies about Josef K., he knew he had done nothing wrong but, one morning, he was arrested.",
            "If you really want to hear about it, the first thing you'll probably want to know is where I was born, and what my lousy childhood was like.",
        ]

    @staticmethod
    def load_from_file(path: str) -> 'PromptBank':
        """Load prompts from file"""
        with open(path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return PromptBank(prompts)
