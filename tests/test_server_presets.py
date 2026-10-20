import server.app as server_app


def test_torch_backend_hides_typed_presets(monkeypatch):
    monkeypatch.setattr(server_app, "USE_MLX", False)
    ids = {p["id"] for p in server_app.list_presets()}
    assert "insurance_claim" not in ids
    assert "smart_home_voice" in ids


def test_mlx_backend_lists_typed_presets(monkeypatch):
    monkeypatch.setattr(server_app, "USE_MLX", True)
    assert "insurance_claim" in {p["id"] for p in server_app.list_presets()}
