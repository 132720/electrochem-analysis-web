from pathlib import Path

import re


TECHNIQUE_TOKENS = {

    "CV": [
        "cv",
        "cyclicvoltammetry",
        "cyclic_voltammetry",
    ],

    "GCD": [
        "gcd",
        "galvanostatic",
        "charge_discharge",
        "chronopotentiometry",
    ],

    "EIS": [
        "eis",
        "impedance",
        "ac_impedance",
    ],
}


# =========================================================
# Temperature from filename
# =========================================================

def infer_temperature_from_filename(
    filename: str,
):

    stem = (
        Path(
            filename
        ).stem
    )

    patterns = [

        r"(-?\d+(?:\.\d+)?)\s*°\s*[Cc]",

        r"(-?\d+(?:\.\d+)?)\s*℃",

        r"(-?\d+(?:\.\d+)?)\s*[_-]?\s*[Cc](?:\b|$)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            stem,
            flags=
                re.IGNORECASE,
        )

        if match:

            try:

                return float(
                    match.group(
                        1
                    )
                )

            except ValueError:

                return None

    return None


# =========================================================
# Group label
# =========================================================

def infer_group_label(
    filename: str,
    technique: str | None = None,
) -> str:

    temperature = (
        infer_temperature_from_filename(
            filename
        )
    )

    if temperature is not None:

        return (
            f"{temperature:g} °C"
        )

    stem = (
        Path(
            filename
        ).stem
    )

    cleaned = (
        stem
    )

    tokens = []

    if technique in TECHNIQUE_TOKENS:

        tokens.extend(
            TECHNIQUE_TOKENS[
                technique
            ]
        )

    else:

        for values in (
            TECHNIQUE_TOKENS.values()
        ):

            tokens.extend(
                values
            )

    for token in tokens:

        cleaned = re.sub(
            rf"(^|[_\-\s]){re.escape(token)}($|[_\-\s])",
            "_",
            cleaned,
            flags=
                re.IGNORECASE,
        )

    cleaned = re.sub(
        r"[_\-\s]+",
        " ",
        cleaned,
    ).strip()

    if cleaned:

        return cleaned

    return stem


# =========================================================
# Group datasets
# =========================================================

def group_datasets(
    datasets,
    labels: list[str],
):

    groups = {}

    for data, label in zip(
        datasets,
        labels,
    ):

        group = groups.setdefault(
            label,
            {
                "CV": [],
                "GCD": [],
                "EIS": [],
            },
        )

        group.setdefault(
            data.technique,
            [],
        ).append(
            data
        )

    return groups
