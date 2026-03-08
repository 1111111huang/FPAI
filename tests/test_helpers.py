from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import generate_match_id


def test_generate_match_id_is_deterministic_for_same_input() -> None:
    match_id_1 = generate_match_id("15/08/2025", "Liverpool", "Bournemouth")
    match_id_2 = generate_match_id("15/08/2025", "Liverpool", "Bournemouth")

    assert match_id_1 == match_id_2


def test_generate_match_id_normalizes_case_and_whitespace() -> None:
    clean = generate_match_id("15/08/2025", "Liverpool", "Bournemouth")
    noisy = generate_match_id(" 15/08/2025 ", " liverpool ", "  BOURNEMOUTH  ")

    assert clean == noisy
