# sidecar-search

A set of CLI build tools for "sidecar indexes" to add semantic search to anything. Unlike RAG solutions like Pinecone, a "sidecar index" is _not_ a vector database: it does not claim to hold the authoritative copy of a text. Rather it is just a fast, small mapping from a query to IDs, ranked by semantic similarity. This enables attaching semantic search to _any_ existing endpoint that enables fetching by ID--including third-party ones.

Originally cleaved off from [abstracts-search](https://github.com/colonelwatch/abstracts-search), a project to enable search through 200M academic publications on OpenAlex while storing nothing but 20GB+ sidecar index, and made generic.

## Installation

Currently, sidecar-search can be installed from source with the following commands.

```bash
git clone https://github.com/colonelwatch/sidecar-search
cd sidecar-search
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Usage

sidecar-search is a CLI application, and after installation, the latest interface can be found by checking the help message.

```bash
sidecar-search --help
```

## Development

The development suite is available as a dependency group, and it can be installed with this editable install command.

```bash
pip install --group dev -e .
```
