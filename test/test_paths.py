from src.paths import project_dir

def test_paths():
    assert project_dir.exists(), f"Project directory {project_dir} does not exist."
    assert project_dir.is_dir(), f"Project directory {project_dir} is not a directory."
    assert (project_dir / "data").exists(), f"Data directory {project_dir / 'data'} does not exist."
    assert (project_dir / "data").is_dir(), f"Data directory {project_dir / 'data'} is not a directory."
    