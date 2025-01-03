install: ## Update dependencies after updating requirements.txt
	@echo "🚀 Install dependencies using uv"
	@uv pip install -r requirements.txt

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := run-dev