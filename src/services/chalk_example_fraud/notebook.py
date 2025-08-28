from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import polars as pl
from chalk.client import ChalkClient
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from chalk.client.response import OnlineQueryResult

# ---------------------------------------------------------------------------


def setup_otel(hyperdx_api_key: str) -> trace.Tracer:
    """Setup OpenTelemetry tracing with HyperDX."""
    trace.set_tracer_provider(TracerProvider())

    otlp_exporter = OTLPSpanExporter(
        endpoint="https://in-otel.hyperdx.io/v1/traces",
        headers={"authorization": hyperdx_api_key},
    )

    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)  # type: ignore[union-attr]

    return trace.get_tracer(__name__)


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
    # 1️⃣ Get environment variables
    client_id: str | None = os.getenv("CHALK_CLIENT_ID")
    client_secret: str | None = os.getenv("CHALK_CLIENT_SECRET")
    hyperdx_api_key: str | None = os.getenv("HYPERDX_API_KEY")

    if client_id is None or client_secret is None:
        msg: str = (
            "CHALK_CLIENT_ID and CHALK_CLIENT_SECRET environment variables must be set"
        )
        raise ValueError(msg)

    if hyperdx_api_key is None:
        msg: str = "HYPERDX_API_KEY environment variable must be set"
        raise ValueError(msg)

    # 2️⃣ Setup OpenTelemetry
    tracer: trace.Tracer = setup_otel(hyperdx_api_key)

    # 3️⃣ Initialise Chalk client
    client: ChalkClient = ChalkClient(
        client_id=client_id,
        client_secret=client_secret,
    )

    # 4️⃣ Load recording IDs
    call_ids_data: pl.DataFrame = pl.read_csv(
        source="data/fathom/call_ids.csv",
        has_header=True,
        schema_overrides={"recording_id": pl.Utf8},
    )
    print("Loaded call IDs data")

    # 5️⃣ Iterate over recordings, query, clean & log
    with tracer.start_as_current_span("fathom_call_processing") as span:
        span.set_attribute("total_recordings", len(call_ids_data))

        for i, recording_id in enumerate(call_ids_data["recording_id"]):
            with tracer.start_as_current_span("process_recording") as recording_span:
                recording_span.set_attribute("recording_id", recording_id)
                recording_span.set_attribute("recording_index", i)

                try:
                    # Query Chalk
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

                    # Log the data as JSON
                    recording_span.add_event(
                        "fathom_call_data",
                        {
                            "recording_id": recording_id,
                            "data": json.dumps(cleaned_dict),
                            "fields_removed_count": 6,
                        },
                    )

                    recording_span.set_attribute("status", "success")
                    print(f"Logged data for recording {recording_id}")

                except Exception as e:
                    recording_span.set_attribute("status", "error")
                    recording_span.set_attribute("error", str(e))
                    recording_span.add_event("error", {"message": str(e)})
                    print(f"Error processing recording {recording_id}: {e}")
                    raise


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
