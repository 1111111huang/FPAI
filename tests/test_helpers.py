from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import generate_match_id


def test_generate_match_id_is_deterministic_for_same_input() -> None:
    match_id_1 = generate_match_id("15/08/2025", "Liverpool", "Bournemouth", "E0")
    match_id_2 = generate_match_id("15/08/2025", "Liverpool", "Bournemouth", "E0")

    assert match_id_1 == match_id_2


def test_generate_match_id_normalizes_case_and_whitespace() -> None:
    clean = generate_match_id("15/08/2025", "Liverpool", "Bournemouth", "E0")
    noisy = generate_match_id(" 15/08/2025 ", " liverpool ", "  BOURNEMOUTH  ", " e0 ")

    assert clean == noisy


def test_generate_match_id_differs_by_league() -> None:
    # US#140: same date + same teams under two different competitions must
    # never collide onto the same match_id.
    e0_id = generate_match_id("15/08/2025", "Liverpool", "Bournemouth", "E0")
    swe_id = generate_match_id("15/08/2025", "Liverpool", "Bournemouth", "SWE")

    assert e0_id != swe_id


def test_generate_match_id_requires_league_argument() -> None:
    with pytest.raises(TypeError):
        generate_match_id("15/08/2025", "Liverpool", "Bournemouth")  # type: ignore[call-arg]
