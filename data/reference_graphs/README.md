# Reference Graphs

This directory contains ground truth causal graphs for comparison.

## File Format (Tetrad Format)

Each reference graph file should follow Tetrad's text format:

```
Graph Nodes:
X1;X2;X3;X4

Graph Edges:
1. X1 --> X2
2. X2 --> X3
3. X3 <-> X4
```

## Edge Types

| Symbol | Meaning | Description |
|--------|---------|-------------|
| `-->` | Directed | A causes B |
| `<->` | Bidirected | Latent confounder between A and B |
| `---` | Undirected | A and B are related (direction unknown) |
| `o->` | PAG circle-arrow | A possibly causes B |
| `o-o` | PAG circle-circle | Possible relationship (direction unknown) |

## Variable Name Mapping

You can use **friendly names** in your reference graphs and map them to actual data column names using `variable_maps.json`.

### How It Works

1. **Reference graph** uses friendly names (e.g., `PolicyRate`, `Consumption`)
2. **variable_maps.json** maps these to data columns (e.g., `FEDFUNDS`, `Consumption_PCE_logdiff`)
3. The comparison tool automatically applies the mapping

### Example

**Reference graph (`monetary_shock_graph_friendly.txt`):**
```
Graph Nodes:
MonetaryShock;PolicyRate;CreditConditions;Consumption;Output

Graph Edges:
1. MonetaryShock --> PolicyRate
2. PolicyRate --> CreditConditions
3. CreditConditions --> Consumption
4. Consumption --> Output
```

**Variable mapping (`variable_maps.json`):**
```json
{
    "monetary_shock": {
        "graph_to_data": {
            "MonetaryShock": "MonetaryShock_RR",
            "PolicyRate": "FEDFUNDS",
            "CreditConditions": "baa_aaa_creditconditions",
            "Consumption": "Consumption_PCE_logdiff",
            "Output": "Output_IP_logdiff"
        }
    }
}
```

## Adding a New Dataset

1. Create a reference graph file here: `your_dataset_graph.txt`

2. (Optional) Add variable mapping to `variable_maps.json`:
```json
{
    "your_dataset": {
        "graph_to_data": {
            "FriendlyName1": "data_column_1",
            "FriendlyName2": "data_column_2"
        }
    }
}
```

3. Add the configuration to `run_comparison.py`:

```python
DATASETS = {
    # ... existing datasets ...
    'your_dataset': {
        'data': 'data/your_data.csv',
        'reference': 'data/reference_graphs/your_dataset_graph.txt',
        'var_map': 'data/reference_graphs/variable_maps.json',
        'description': 'Description of your dataset'
    },
}
```

4. Run comparison:
```bash
python run_comparison.py --dataset your_dataset
```

## Current Datasets

### 1. Monetary Policy Transmission DAG (Romer & Romer Style)
File: `monetary_shock_graph.txt`

**Variables (10):**
- MonetaryShock_RR - Monetary Shock (RR)
- FEDFUNDS - Policy Rate
- t_bill_inflationexpectations - Inflation Expectations
- baa_aaa_creditconditions - Credit Conditions 
- assetprice_sp500_logdiff - Asset Prices 
- RNUSBIS_logdiff - Exchange Rate
- Investment_logdiff - Investment
- Consumption_PCE_logdiff - Consumption 
- Output_IP_logdiff - Output 
- NX GDP RATIO PCT - Net Exports

**Edges (17):** Full transmission mechanism from monetary shock through policy rate, credit conditions, asset prices, and real economy.

### 2. Housing Wealth Transmission Mechanism
File: `housing_graph.txt`

**Variables (5):**
- MORTGAGE30US - Mortgage Rates
- CreditSupply_logdiff - Credit Supply
- HousePrice_logdiff - House Prices
- Consumption_logdiff - Consumption 
- Output_logdiff - Output 

**Edges (4):** Mortgage rates and credit supply affect house prices, which affect consumption and output.

