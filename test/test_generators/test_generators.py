import pytest
from faker import Faker
from synthetic_data_generation.generators import (create_name_surname,
                                                  generate_country,
                                                  Gender)


@pytest.fixture
def faker():
    fake = Faker()
    return fake


def test_create_name_surname(faker):
    names = create_name_surname(Gender.male, faker)
    assert len(names) == 2


def test_generate_country(faker):
    countries = generate_country(samples=2,
                                 fake=faker)
    assert len(countries) == 2
