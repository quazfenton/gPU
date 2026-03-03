# Cost Tracking Guide

**Version:** 1.0.0  
**Last Updated:** March 3, 2026

---

## Overview

The Notebook ML Orchestrator includes comprehensive cost tracking across all ML backends. This guide explains how costs are calculated, monitored, and optimized.

---

## Cost Calculation

### Backend Pricing

| Backend | GPU Type | Cost/Hour | Free Tier |
|---------|----------|-----------|-----------|
| Modal | T4 | $0.60 | Limited testing |
| Modal | A10G | $1.10 | Limited testing |
| Modal | A100 | $4.00 | No |
| HuggingFace | T4 | $0.00 | Yes (1000 req/hr) |
| Kaggle | T4 x2 | $0.00 | Yes (30 hrs/week) |
| Colab | T4 | $0.00 | Yes (limits vary) |

### Cost Formula

```
Job Cost = Backend Rate × Duration (hours)

Minimum charge: 0.01 hours (36 seconds)
```

### Example Calculations

**Example 1: Modal A10G Job (5 minutes)**
```
Duration: 5 minutes = 0.083 hours
Rate: $1.10/hour
Cost: 0.083 × $1.10 = $0.091
```

**Example 2: Kaggle GPU Job (2 hours)**
```
Duration: 2 hours
Rate: $0.00 (free tier)
Cost: $0.00
(Counts against 30 hour/week quota)
```

**Example 3: HuggingFace Inference (1000 requests)**
```
Duration: N/A (per-request)
Rate: $0.00 (free tier)
Cost: $0.00
(Counts against 1000 requests/hour)
```

---

## Using Cost Tracking Dashboard

### Access Dashboard

1. Open GUI: `http://localhost:7860`
2. Navigate to **Cost Tracking** tab
3. View real-time cost data

### Dashboard Features

#### Summary Cards
- **Total Cost**: Total spend across all jobs
- **Total Jobs**: Number of jobs executed
- **Avg Cost/Job**: Average cost per job
- **Estimated Monthly**: Projected monthly cost

#### Cost by Backend
- Bar chart showing cost distribution
- Table with breakdown by backend
- Percentage of total spend

#### Cost by Template
- Horizontal bar chart (top 10)
- Table with cost per template
- Identify expensive templates

#### Budget Alerts
- Set monthly budget
- Configure alert threshold (%)
- Track budget consumption

#### Expensive Jobs
- Top 10 most expensive jobs
- Last 7 days by default
- Identify cost optimization opportunities

### Time Ranges

- **Last 24 Hours**: Recent activity
- **Last 7 Days**: Weekly view
- **Last 30 Days**: Monthly view
- **All Time**: Historical total

---

## Cost Optimization Strategies

### 1. Use Free Tiers When Possible

**Strategy:** Route jobs to free tier backends first

```python
# In backend router, prefer free tiers
routing_strategy = "cost-optimized"  # Default
prefer_free_tier = True
```

**Savings:** Up to 100% for eligible workloads

### 2. Right-Size GPU Requirements

**Strategy:** Use appropriate GPU for workload

| Workload Type | Recommended GPU | Cost/Hour |
|--------------|-----------------|-----------|
| Inference (small models) | CPU or T4 | $0.00-$0.60 |
| Inference (large models) | A10G | $1.10 |
| Training (small) | T4 or A10G | $0.60-$1.10 |
| Training (large) | A100 | $4.00 |

**Savings:** 50-85% by avoiding over-provisioning

### 3. Batch Processing

**Strategy:** Combine multiple small jobs

```python
# Instead of 10 separate jobs:
for item in items:
    submit_job(item)  # 10 × minimum charge

# Use batch processing:
submit_batch(items)  # Single job, lower cost
```

**Savings:** 30-50% for small jobs

### 4. Monitor Job Duration

**Strategy:** Set appropriate timeouts

```python
# Don't use 1 hour timeout for 5-minute job
job = Job(
    template="image-classification",
    timeout=300  # 5 minutes, not 3600
)
```

**Savings:** Prevents runaway costs

### 5. Use Cost Tracking Dashboard

**Strategy:** Review costs weekly

- Check "Cost by Backend" for optimization opportunities
- Review "Expensive Jobs" for anomalies
- Set budget alerts for early warning

---

## Budget Management

### Setting Budget

1. Go to **Cost Tracking** tab
2. Enter monthly budget (e.g., $100)
3. Set alert threshold (e.g., 80%)
4. Click **Save Budget Settings**

### Alert Notifications

**Current Implementation:**
- Visual indicators in dashboard
- Budget status display

**Future Enhancements:**
- Email alerts at 80%, 90%, 100%
- Slack/Discord notifications
- Automatic job throttling at limit

### Budget Best Practices

**For Development:**
- Budget: $50-100/month
- Alert at: 80%
- Review: Weekly

**For Production:**
- Budget: Based on workload
- Alert at: 70%, 90%, 100%
- Review: Daily

---

## Cost Export

### Export Data

```python
from notebook_ml_orchestrator.security import SecurityLogger

logger = SecurityLogger()

# Export cost events to CSV
csv_data = logger.export_events(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    event_types=['job.completed'],
    output_format='csv'
)

# Save to file
with open('cost_report.csv', 'w') as f:
    f.write(csv_data)
```

### Export Formats

- **JSON**: Machine-readable, full metadata
- **CSV**: Spreadsheet-compatible
- **CEF**: SIEM integration (Splunk)
- **LEEF**: SIEM integration (QRadar)

---

## API Access

### Get Cost Data Programmatically

```python
from notebook_ml_orchestrator.core.backend_router import MultiBackendRouter

router = MultiBackendRouter()

# Get cost optimizer
cost_optimizer = router.cost_optimizer

# Get total cost
total = cost_optimizer.get_total_cost()

# Get cost by backend
modal_cost = cost_optimizer.get_total_cost('modal')
```

### Future API Endpoints

```bash
# Get current costs
GET /api/costs/current

# Get cost breakdown
GET /api/costs/breakdown?backend=modal&template=image-classification

# Set budget
POST /api/costs/budget
{
  "monthly_limit": 100,
  "alert_threshold": 80
}
```

---

## Troubleshooting

### Issue: Costs Not Showing

**Symptoms:** Dashboard shows $0.00

**Solutions:**
1. Check jobs have been executed
2. Verify backend is registered
3. Refresh dashboard (click 🔄 Refresh)
4. Check time range filter

### Issue: Inaccurate Cost Estimates

**Symptoms:** Costs don't match expectations

**Solutions:**
1. Verify backend pricing is current
2. Check job duration calculation
3. Review backend selection (auto vs manual)
4. Consider minimum charge (0.01 hours)

### Issue: Budget Alerts Not Working

**Symptoms:** No alerts when budget exceeded

**Solutions:**
1. Verify budget is saved
2. Check alert threshold setting
3. Review notification settings (future feature)
4. Monitor dashboard manually for now

---

## Cost Optimization Checklist

### Daily
- [ ] Check dashboard for anomalies
- [ ] Review failed jobs (wasted costs)
- [ ] Monitor queue lengths

### Weekly
- [ ] Review cost by backend
- [ ] Review cost by template
- [ ] Check budget consumption
- [ ] Identify optimization opportunities

### Monthly
- [ ] Compare actual vs estimated costs
- [ ] Review budget settings
- [ ] Adjust routing strategy if needed
- [ ] Plan for next month's workload

---

## Future Enhancements

### Planned Features (Phase 6+)

1. **Real-time Budget Alerts**
   - Email notifications
   - Slack/Discord integration
   - Automatic job throttling

2. **Cost Forecasting**
   - ML-based predictions
   - Trend analysis
   - Anomaly detection

3. **Reserved Capacity**
   - Reserved instance support
   - Spot instance integration
   - Cost commitment discounts

4. **Multi-Currency Support**
   - USD, EUR, GBP, JPY
   - Exchange rate updates
   - Regional pricing

5. **Chargeback/Showback**
   - Cost allocation by user
   - Department billing
   - Invoice generation

---

## References

### Backend Pricing Pages
- [Modal Pricing](https://modal.com/pricing)
- [HuggingFace Pricing](https://huggingface.co/pricing)
- [Kaggle Limits](https://www.kaggle.com/docs/faq#resourceLimits)
- [Google Colab Pricing](https://colab.research.google.com/subscribe)

### Related Documentation
- [Production Deployment Checklist](PRODUCTION_DEPLOYMENT_CHECKLIST.md)
- [Security Implementation Guide](PHASE3_SECURITY_COMPLETE.md)
- [Final Implementation Report](FINAL_IMPLEMENTATION_REPORT.md)

---

**Questions?** See [FINAL_IMPLEMENTATION_REPORT.md](FINAL_IMPLEMENTATION_REPORT.md) for complete documentation.
