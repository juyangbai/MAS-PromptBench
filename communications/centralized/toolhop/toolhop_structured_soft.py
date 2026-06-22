from communications.communication_formats import install_proxy, cli_main

install_proxy(globals(), topology="centralized", dataset="toolhop", fmt="structured_soft")

if __name__ == "__main__":
    raise SystemExit(cli_main(globals()))
