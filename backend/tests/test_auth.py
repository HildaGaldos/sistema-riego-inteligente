from backend.app.auth import create_token, decode_token, hash_password, verify_password


def test_password_and_token_roundtrip():
    stored = hash_password("secret")
    assert verify_password("secret", stored)
    assert not verify_password("wrong", stored)
    token = create_token("alice", "test-secret", 5)
    assert decode_token(token, "test-secret")["sub"] == "alice"
