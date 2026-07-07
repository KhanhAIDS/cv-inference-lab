import pytest
from smoke_fire.data.audit_dfire import check_split

@pytest.fixture
def dummy_split_dir(tmp_path):
    split_dir = tmp_path / "train"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    
    # Create valid pair
    (images_dir / "valid1.jpg").touch()
    with open(labels_dir / "valid1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
        
    # Create empty label (negative)
    (images_dir / "empty1.jpg").touch()
    (labels_dir / "empty1.txt").touch()
    
    # Create invalid class
    (images_dir / "invalid1.jpg").touch()
    with open(labels_dir / "invalid1.txt", "w") as f:
        f.write("2 0.5 0.5 0.2 0.2\n")
        
    # Create malformed row
    (images_dir / "malformed1.jpg").touch()
    with open(labels_dir / "malformed1.txt", "w") as f:
        f.write("0 0.5 0.5\nnot_a_number 0.2\n")
        
    return split_dir

def test_audit_dfire_label_parser(dummy_split_dir):
    report = check_split(dummy_split_dir)
    assert report is not None
    assert report["total_images"] == 4
    assert report["total_labels"] == 4
    assert report["empty_labels"] == 1
    # invalid1 has invalid class id "2", malformed1 has "not_a_number"
    assert report["invalid_labels"] == 2
    assert report["class_counts"][0] == 2 # 1 from valid1, 1 from the valid line in malformed1
    assert report["class_counts"][1] == 0
