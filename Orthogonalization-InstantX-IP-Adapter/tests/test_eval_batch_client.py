from pathlib import Path

import eval_batch_client


def test_evaluation_download_temporary_name_stays_short() -> None:
    destination = (
        Path("C:/")
        / ("x" * 180)
        / "_metadata"
        / "i"
        / "official_animal_028"
        / "seed_0045.json"
    )
    temporary = eval_batch_client._download_temporary_path(destination)

    assert temporary.parent == destination.parent
    assert temporary.name.startswith(".d-")
    assert temporary.name.endswith(".tmp")
    assert len(temporary.name) == 19
    assert len(str(temporary)) - len(str(destination)) == 5
