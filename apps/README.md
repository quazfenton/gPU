# Modal App Library

Reusable serverless function templates for Modal.com deployment.

## Available Apps

| App | GPU | Description |
|-----|-----|-------------|
| image-classifier | T4 | ResNet50 image classification |
| text-generator | T4 | GPT-2 text generation |
| web-scraper | None | BeautifulSoup web scraping |
| batch-processor | None | Pandas batch processing |
| scheduled-task | None | Hourly cron job example |

## Quick Commands

```bash
# List all apps
python3 runna.py app-list

# View app code
python3 runna.py app-show image-classifier

# Deploy app
python3 runna.py app-deploy image-classifier

# Add custom app
python3 runna.py app-add my-app --file my_app.py --gpu T4
```

## Files

- `library.json` - App index and metadata
- `*.py` - App templates
- `*.md` - Documentation from Modal examples

## See Also

- [APP_LIBRARY.md](../APP_LIBRARY.md) - Complete documentation
- [MODAL_GUIDE.md](../MODAL_GUIDE.md) - Deployment guide
