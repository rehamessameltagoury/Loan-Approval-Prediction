test:
	pytest tests/

quality_checks:
	isort .
	black .

