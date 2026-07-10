"""Console entrypoint for FoxNovo."""


def main(*args, **kwargs):
    from .api_denovo import main as main_impl

    return main_impl(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
