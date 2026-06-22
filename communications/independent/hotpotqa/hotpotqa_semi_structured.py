from communications.communication_formats import install_proxy, cli_main

install_proxy(globals(), topology="independent", dataset="hotpotqa", fmt="semi_structured")


if __name__ == "__main__":
    raise SystemExit(cli_main(globals()))
