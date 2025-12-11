# US Indirect Tax Q&A Sample

Evaluates LLM responses to US sales/use tax questions using GoTaglio.

## Files

- `tax.py` - Pipeline definition (prepare → infer → extract → assess)
- `tax.ipynb` - Interactive notebook
- `data/cases.yaml` - Test cases (16 cases covering software, food, medical, etc.)
- `data/template.txt` - System prompt for tax expert persona

## Usage

Open `tax.ipynb` and run cells. The notebook:
1. Loads cases from `data/cases.yaml`
2. Runs with mock model (`perfect`) or real LLM
3. Saves results to `logs/`

## Model Setup

Configure models in `/workspaces/gotaglio/models.json` and API keys in `.credentials.json`.

## Evaluation

The `assess` stage checks:
- `taxable` - exact match (true/false)
- `rate` - within 1% tolerance
