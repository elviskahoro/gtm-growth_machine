from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from chalk.client import ChalkClient

if TYPE_CHECKING:
    from chalk.client.response import OnlineQueryResult

# ---------------------------------------------------------------------------


def convert_datetimes(obj: Any) -> Any:  # trunk-ignore(ruff/ANN401)
    """Recursively convert datetime objects to ISO strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: convert_datetimes(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_datetimes(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------


def main() -> None:
    # 2️⃣ Initialise Chalk client
    client_id: str | None = os.getenv("CHALK_CLIENT_ID")
    client_secret: str | None = os.getenv("CHALK_CLIENT_SECRET")

    if client_id is None or client_secret is None:
        msg: str = (
            "CHALK_CLIENT_ID and CHALK_CLIENT_SECRET environment variables must be set"
        )
        raise ValueError(msg)
    client: ChalkClient = ChalkClient(
        client_id=client_id,
        client_secret=client_secret,
    )

    # 3️⃣ Load recording IDs
    call_ids_data: pl.DataFrame = pl.read_csv(
        source="data/fathom/call_ids.csv",
        has_header=True,
        schema_overrides={"recording_id": pl.Utf8},
    )
    print("Loaded call IDs data")

    # 4️⃣ Create data directory
    data_dir: Path = Path.cwd() / "out/fathom_call-backfill/"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 5️⃣ Iterate over recordings, query, clean & save
    for recording_id in call_ids_data["recording_id"]:
        query: OnlineQueryResult = client.query(
            input={"fathom_call.id": recording_id},
        )
        data: dict[str, Any] = query.to_dict()

        # Remove unwanted fields
        for key in [
            "fathom_call.llm_call_summary_general",
            "fathom_call.llm_call_summary_sales",
            "fathom_call.llm_call_summary_marketing",
            "fathom_call.llm_call_type",
            "fathom_call.llm_call_insights_sales",
            "fathom_call.transcript_plaintext_list",
        ]:
            data.pop(key, None)

        cleaned_dict: Any = convert_datetimes(data)

        # Write to JSON file
        file_path: Path = data_dir / f"{recording_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(cleaned_dict, f, indent=None, ensure_ascii=False)

        print(f"Saved {file_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
