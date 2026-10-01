def test_feature_schema_matches_core():
    from core.schema import FEATURE_NAMES
    assert isinstance(FEATURE_NAMES, list)
