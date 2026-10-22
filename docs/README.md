# Docs site

Everything for the documentation site lives in this folder:

```text
docs/
├── mkdocs.yml          site config and navigation (33 pages)
├── requirements.txt    mkdocs + pymdown-extensions (pinned)
├── content/            the Markdown pages; topology diagrams in content/_snippets/
│                       (reference/ is the configuration and CLI reference the READMEs link to)
└── theme/              the custom theme (main.html, docs.css, docs.js), make_diagrams.py
                        and AUTHORING.md (writing guide)
```

One file sits outside it because GitHub only reads workflows from a fixed path:
`.github/workflows/docs.yml` (builds on push to `main` and publishes to `webpage:/docs`).

Preview from the repository root:

```bash
pip install -r docs/requirements.txt
mkdocs serve -f docs/mkdocs.yml            # http://127.0.0.1:8000, reloads on save
mkdocs build --strict -f docs/mkdocs.yml   # fails on broken links or anchors, same as CI
```

The workflow publishes the site to `https://juyangbai.github.io/MAS-PromptBench/docs/`.
It replaces only the `docs/` folder on the `webpage` branch, so `index.html` and `quickstart.html`
stay as they are.

## DeepWiki

1. Open https://deepwiki.com, submit `https://github.com/juyangbai/MAS-PromptBench` and wait for the first build.
2. The badge at the top of the repository `README.md` links to the wiki and keeps it refreshed.

## Editing

- Add a page: create the `.md` file under `docs/content/` and add it to `nav` in `docs/mkdocs.yml`.
- The repository READMEs link into `docs/content/` with relative paths and anchors; find them with
  `grep -rn 'docs/content/' --include='*.md' .` before you rename a page or a heading.
- Change a topology diagram: edit `docs/theme/make_diagrams.py` and run `python docs/theme/make_diagrams.py`.
- Follow `docs/theme/AUTHORING.md` for voice, page shape and the Markdown features the theme styles
  (fact strips, card grids, callouts, tabs, code titles).

## Page transition

`docs/theme/assets/mpb-transition.css` and `mpb-transition.js` add the "Short glide, ordered" transition
between the docs and the project page. The project page on `webpage` loads the same two files from
`static/css/` and `static/js/`.
