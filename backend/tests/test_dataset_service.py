from io import BytesIO
from types import SimpleNamespace

from backend.app.services.dataset_service import save_uploaded_dataset


def test_manual_upload_validates_aliases_without_synthetic_fallback(tmp_path):
    csv = b"Temperature,Humidity,MOI,Soil Type,Seedling Stage,Result\n20,60,25,Black Soil,Germination,0\n32,40,80,Red Soil,Vegetative,1\n"
    uploaded = SimpleNamespace(filename="smart-agriculture.csv", file=BytesIO(csv))
    result = save_uploaded_dataset(uploaded, tmp_path)
    assert result["quality"]["rows"] == 2
    assert result["filename"].endswith(".csv")
