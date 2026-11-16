import pytest

from explain_service import ExplainUnavailableError, select_backend
from settings import Settings, read_env_file

pytest.importorskip("catboost")

from dosing_model import age_to_bucket, vkorc1_to_consensus


def make_settings(**overrides) -> Settings:
    values = {
        "openrouter_api_key": "",
        "openrouter_model": "default/model",
        "hf_token": "",
        "hf_medgemma_model": "google/medgemma-test",
    }
    return Settings(**{**values, **overrides})


@pytest.mark.parametrize(
    ("age", "bucket"),
    [(19, "10 - 19"), (20, "20 - 29"), (65, "60 - 69"), (89, "80 - 89"), (90, "90+"), (101, "90+")],
)
def test_age_to_bucket(age, bucket):
    assert age_to_bucket(age) == bucket


@pytest.mark.parametrize(
    ("code", "consensus"),
    [("GG", "G/G"), ("AG", "A/G"), ("AA", "A/A"), ("xx", "missing"), ("unknown", "missing")],
)
def test_vkorc1_to_consensus(code, consensus):
    assert vkorc1_to_consensus(code) == consensus


def test_explain_provider_prefers_huggingface():
    settings = make_settings(hf_token="hf", openrouter_api_key="or")
    assert settings.explain_provider == "huggingface (medgemma)"
    assert make_settings(openrouter_api_key="or").explain_provider == "openrouter"
    assert make_settings().explain_provider == "none (deterministic fallback only)"


def test_hugging_face_backend_only_accepts_medgemma_models():
    settings = make_settings(hf_token="hf")
    assert select_backend(settings, "google/MedGemma-4b").model == "google/MedGemma-4b"
    assert select_backend(settings, "other/model").model == "google/medgemma-test"
    assert select_backend(settings, None).provider == "huggingface"


def test_openrouter_backend_uses_requested_or_default_model():
    settings = make_settings(openrouter_api_key="or")
    assert select_backend(settings, "x/y").model == "x/y"
    assert select_backend(settings, None).model == "default/model"


def test_no_credentials_raises():
    with pytest.raises(ExplainUnavailableError):
        select_backend(make_settings(), None)


def test_read_env_file_ignores_comments_and_blank_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# note\n\nKEY=value\nOTHER = spaced \nbroken line\n")
    assert read_env_file(env) == {"KEY": "value", "OTHER": "spaced"}
    assert read_env_file(tmp_path / "missing.env") == {}
