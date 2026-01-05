# HPO with Modal

## Beware, Here Be Dragons

Running hyperparameter optimization on Modal with ultrathink compute is **extremely expensive**. A single HPO sweep can burn through significant cloud credits in minutes.

### Before You Run Anything

1. **Set budget limits** in your Modal dashboard
2. **Start with small trial counts** (e.g., 3-5 trials, not 100)
3. **Use cheaper GPU tiers first** (T4) before scaling to A100s
4. **Monitor costs in real-time** during runs

### Cost Estimation

| GPU Type | Approx. Cost/Hour | HPO Risk Level |
|----------|-------------------|----------------|
| T4       | ~$0.60           | Low            |
| A10G     | ~$1.10           | Medium         |
| A100-40G | ~$3.00           | High           |
| A100-80G | ~$4.50           | Very High      |

A 100-trial HPO sweep with 30-minute trials on A100-80G = **~$225**

### Recommended Workflow

1. Debug locally with CPU/small data
2. Run 3 trials on T4 to validate setup
3. Scale up incrementally with budget caps
