# trunk-ignore-all(ruff/PLW0603)
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import uuid6
from dotenv import load_dotenv
from faker import Faker
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import declarative_base, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

RANDOM_NUMBER_GENERATOR_SEED: int = 47
DATABASE_NAME = "ecommerce"
NROWS: int | None = None
SKIPROWS: int | None = None

# Constants for magic numbers
PERCENT_THRESHOLD_ADDITIONAL_INTERACTIONS = 20
PERCENT_THRESHOLD_NO_REVIEW_INTERACTIONS = 15
MIN_ROWS_TO_PROCESS = 1000

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
)

load_dotenv()
DATABASE_USER = os.environ["DB_USER"]
DATABASE_PASSWORD = os.environ["DB_PASSWORD"]
DATABASE_HOST = os.environ["DB_HOST"]

url = URL.create(
    drivername="postgresql",
    username=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    port=5432,
    database=DATABASE_NAME,
)

engine = create_engine(
    url=url,
    pool_size=20,
    future=True,
)

SessionMaker = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

f = Faker()
Faker.seed(0)
fake = Faker()

Base = declarative_base()


class MarketplaceReview(Base):
    """Marketplace reviews are generated from the Amazon book reviews dataset.
    https://www.kaggle.com/datasets/machharavikiran/amazon-reviews.
    """

    __tablename__ = "marketplace_reviews"

    id = Column(String, nullable=False, primary_key=True)
    at = Column(DateTime, nullable=False)
    marketplace = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    product_id = Column(String, nullable=False)
    product_parent = Column(String, nullable=False)
    product_title = Column(String, nullable=False)
    product_category = Column(String, nullable=False)
    star_rating = Column(Integer, nullable=False)
    helpful_votes = Column(Integer, nullable=True)
    total_votes = Column(Integer, nullable=True)
    vine = Column(String, nullable=False)
    verified_purchase = Column(String, nullable=False)
    review_headline = Column(String, nullable=True)
    review_body = Column(String, nullable=True)

    interaction_id = Column(String, nullable=True)


class MarketplaceInteraction(Base):
    __tablename__ = "marketplace_interactions"

    id = Column(String, primary_key=True, index=True)
    at = Column(DateTime, nullable=False)
    seller_id = Column(
        String,
        nullable=False,
    )
    user_id = Column(String, nullable=False)
    interaction_type = Column(String, nullable=False)
    product_id = Column(String, nullable=False)


# ----------------------------------------------------------------------------
def get_book_reviews_from_disk(
    nrows: int | None,
    skiprows: int | None,
) -> pd.DataFrame:
    path: Path = (
        Path.cwd()
        / "src/marketplace/data/workspace/marketplace-amazon/amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv"
    )
    skiprows_array: list[int] | None = None
    if skiprows:
        # noinspection PyTypeChecker
        skiprows_array = list(
            range(
                1,
                skiprows,
            ),
        )

    return pd.read_csv(
        filepath_or_buffer=path,
        header=0,
        sep="\t",
        on_bad_lines="skip",
        nrows=nrows,
        skiprows=skiprows_array,
    )


def get_user_ids_from_disk(
    nrows: int | None,
    skip_rows: int | None,
) -> pd.DataFrame:
    path: Path = Path.cwd() / "src/marketplace/data/workspace/tables/base_users.csv"
    return pd.read_csv(
        filepath_or_buffer=path,
        header=0,
        sep=",",
        on_bad_lines="skip",
        nrows=nrows,
        skiprows=skip_rows,
    )


def get_seller_ids_from_disk(
    nrows: int | None,
    skiprows: int | None,
) -> pd.DataFrame:
    path: Path = (
        Path.cwd() / "src/marketplace/data/workspace/tables/marketplace_sellers.csv"
    )
    return pd.read_csv(
        filepath_or_buffer=path,
        header=0,
        sep=",",
        on_bad_lines="skip",
        nrows=nrows,
        skiprows=skiprows,
    )


# ----------------------------------------------------------------------------
def generate_data(
    row: pd.Series,
    user_ids: pd.Series | pd.DataFrame,
    seller_ids: pd.Series,
    product_ids: pd.Series,
    random_number_generator: np.random.Generator,
) -> Generator:
    selected_user_ids = {}

    def gen_from_selection(
        row: pd.Series,
        uuid_dict: dict[str, str],
        data_column_for_selection: str,
        data_to_randomly_select_replacement_from: pd.Series | pd.DataFrame,
    ) -> str:
        nonlocal random_number_generator
        new_id: str
        old_id: str = str(row[data_column_for_selection])
        if old_id in uuid_dict:
            return uuid_dict[old_id]

        # noinspection PyTypeChecker,PydanticTypeChecker
        new_id = random_number_generator.choice(
            data_to_randomly_select_replacement_from,
        )[
            0
        ]  # need to extract the string since the returned choice is still a np.array
        uuid_dict[old_id] = new_id
        return new_id

    def gen_review_at() -> datetime:
        nonlocal random_number_generator
        return fake.date_time_between(
            start_date="-60d",
            end_date="now",
        )

    def gen_interaction_timestamps() -> tuple[datetime, datetime]:
        minimum_review_lag_in_minutes: int = 4700
        delta_review_lag_in_minutes: int = 47000
        review_lag_in_minutes: int = int(
            np.ceil(
                np.abs(  # Ensure duration is positive
                    random_number_generator.integers(
                        low=minimum_review_lag_in_minutes,
                        high=minimum_review_lag_in_minutes
                        + delta_review_lag_in_minutes,
                    ),
                ),
            ).astype(int),
        )
        review_interaction_at: datetime = review_at + timedelta(
            minutes=(-1 * review_lag_in_minutes),
        )
        order_placed_delta: int = int(
            np.ceil(
                np.abs(  # Ensure duration is positive
                    random_number_generator.integers(
                        low=review_lag_in_minutes,
                        high=review_lag_in_minutes + delta_review_lag_in_minutes,
                    ),
                ),
            ).astype(int),
        )
        order_placed_interaction_at: datetime = review_interaction_at + timedelta(
            minutes=(-1 * order_placed_delta),
        )
        return review_interaction_at, order_placed_interaction_at

    def gen_marketplace_interactions_with_review(
        user_id: str,
        seller_id: str,
        interaction_id_review: str,
        review_interaction_at: datetime,
        order_placed_interaction_at: datetime,
    ) -> Generator[MarketplaceInteraction, None, None]:
        yield MarketplaceInteraction(
            id=interaction_id_review,
            at=review_interaction_at,
            seller_id=seller_id,
            user_id=user_id,
            interaction_type="feedbackAndReviews",
            product_id=row["product_id"],
        )

        yield MarketplaceInteraction(
            id=str(uuid6.uuid6()),
            at=order_placed_interaction_at,
            seller_id=seller_id,
            user_id=user_id,
            interaction_type="orderPlacement",
            product_id=row["product_id"],
        )

        product_inquiry_delta: int = int(
            np.ceil(
                np.abs(  # Ensure duration is positive
                    random_number_generator.integers(
                        low=2500,
                        high=60000,
                    ),
                ),
            ).astype(int),
        )
        product_inquiry_interaction_at = order_placed_interaction_at + timedelta(
            minutes=(-1 * product_inquiry_delta),
        )
        yield MarketplaceInteraction(
            id=str(uuid6.uuid6()),
            at=product_inquiry_interaction_at,
            seller_id=seller_id,
            user_id=user_id,
            interaction_type="productInquiry",
            product_id=row["product_id"],
        )

    def gen_marketplace_interactions_with_no_review(
        order_placed_interaction_at: datetime,
        seller_id: str,
        user_id: str,
        product_ids: pd.Series,
    ) -> Generator[MarketplaceInteraction, None, None]:
        product_inquiry_delta: int = int(
            np.ceil(
                np.abs(  # Ensure duration is positive
                    random_number_generator.integers(
                        low=300,
                        high=47000,
                    ),
                ),
            ).astype(int),
        )
        product_inquiry_interaction_at = order_placed_interaction_at + timedelta(
            minutes=(-1 * product_inquiry_delta),
        )
        random_product_id: str = str(random_number_generator.choice(product_ids))
        yield MarketplaceInteraction(
            id=str(uuid6.uuid6()),
            at=product_inquiry_interaction_at,
            seller_id=seller_id,
            user_id=user_id,
            interaction_type="productInquiry",
            product_id=random_product_id,
        )
        if (
            random_number_generator.uniform(0, 100)
            < PERCENT_THRESHOLD_ADDITIONAL_INTERACTIONS
        ):
            yield MarketplaceInteraction(
                id=str(uuid6.uuid6()),
                at=product_inquiry_interaction_at,
                seller_id=seller_id,
                user_id=user_id,
                interaction_type="orderPlacement",
                product_id=random_product_id,
            )

    review_at: datetime = gen_review_at()
    user_id: str = gen_from_selection(
        row=row,
        uuid_dict=selected_user_ids,
        data_column_for_selection="customer_id",
        data_to_randomly_select_replacement_from=user_ids,
    )
    interaction_id_review: str = str(uuid6.uuid6())
    yield MarketplaceReview(
        id=str(uuid6.uuid6()),
        user_id=user_id,
        at=review_at,
        interaction_id=interaction_id_review,
        marketplace=row["marketplace"],
        product_id=row["product_id"],
        product_parent=row["product_parent"],
        product_title=row["product_title"],
        product_category=row["product_category"],
        star_rating=row["star_rating"],
        helpful_votes=row["helpful_votes"],
        total_votes=row["total_votes"],
        vine=row["vine"],
        verified_purchase=row["verified_purchase"],
        review_headline=row["review_headline"],
        review_body=row["review_body"],
    )

    review_interaction_at: datetime
    order_placed_interaction_at: datetime
    seller_id: str = str(random_number_generator.choice(seller_ids))
    review_interaction_at, order_placed_interaction_at = gen_interaction_timestamps()
    for marketplace_interaction in gen_marketplace_interactions_with_review(
        interaction_id_review=interaction_id_review,
        order_placed_interaction_at=order_placed_interaction_at,
        review_interaction_at=review_interaction_at,
        user_id=user_id,
        seller_id=seller_id,
    ):
        yield marketplace_interaction

    if (
        random_number_generator.uniform(0, 100)
        < PERCENT_THRESHOLD_NO_REVIEW_INTERACTIONS
    ):
        for marketplace_interaction in gen_marketplace_interactions_with_no_review(
            order_placed_interaction_at=order_placed_interaction_at,
            user_id=user_id,
            seller_id=seller_id,
            product_ids=product_ids,
        ):
            yield marketplace_interaction


# ----------------------------------------------------------------------------
def add_generator(
    g: Generator,
    sql_session,  # trunk-ignore(ruff/ANN001)
) -> None:
    sql_session.add_all(list(g))
    sql_session.commit()


def make_tables(
    *ts,  # trunk-ignore(ruff/ANN002)
) -> None:
    for t in ts:
        t.__table__.drop(engine, checkfirst=True)
        t.__table__.create(engine, checkfirst=True)


if __name__ == "__main__":
    count: int = SKIPROWS if SKIPROWS else 0
    with SessionMaker() as sql_session:

        def reviews_from_dataframe() -> None:
            global count
            # make_tables(MarketplaceReview, MarketplaceInteraction) # clears table
            row_identifier_column: str = "product_title"
            random_number_generator: np.random.Generator = np.random.default_rng(
                seed=RANDOM_NUMBER_GENERATOR_SEED,
            )

            user_ids = get_user_ids_from_disk(
                nrows=NROWS,
                skip_rows=None,
            )
            seller_ids = get_seller_ids_from_disk(
                nrows=NROWS,
                skiprows=None,
            )["id"]
            reviews = get_book_reviews_from_disk(
                nrows=NROWS,
                skiprows=SKIPROWS,
            )
            for _, row in reviews.iterrows():
                count += 1
                print(
                    f"{count:06d}::{row[row_identifier_column]}",
                )
                if count < MIN_ROWS_TO_PROCESS:
                    continue

                add_generator(
                    generate_data(
                        row=row,
                        user_ids=user_ids,
                        seller_ids=seller_ids,
                        product_ids=reviews["product_id"],
                        random_number_generator=random_number_generator,
                    ),
                    sql_session,
                )

        reviews_from_dataframe()
