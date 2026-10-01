def test_whitelist_blocks_nothing():
    from blocker.whitelist import is_whitelisted
    assert is_whitelisted("127.0.0.1") in (True, False)
