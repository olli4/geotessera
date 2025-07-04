import importlib.resources
import pooch

def main():
  version="v1"
  POOCH = pooch.create(
    path=pooch.os_cache("geotessera"),
    base_url="https://dl-1.tessera.wiki/{version}/global_0.1_degree_representation/",
    version=version,
    registry=None,
  )

  with importlib.resources.open_text("geotessera", "registry_2024.txt") as registry_file:
    POOCH.load_registry(registry_file)

  fname=POOCH.fetch(
    "2024/grid_0.15_52.05/grid_0.15_52.05.npy",
    progressbar=True
  )
  print(fname)

if __name__ == "__main__":
    main()
