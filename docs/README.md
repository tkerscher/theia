# Documentation ReadMe

The documentation is built using [zensical](https://zensical.org/)
and [mkdocstrings](https://mkdocstrings.github.io/) to extract API from source
code. All required dependencies are listed in `requirements.txt`, which can be
used with `pip`.

To build the documentation, from the project's root directory run:

```bash
zensical build
```

For a live preview with hot-reload you can instead run:

```bash
zensical serve
```
