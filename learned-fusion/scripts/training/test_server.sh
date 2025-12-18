curl -v \
  -X POST http://localhost:3031/infer \
  -H "Content-Type: text/plain" \
  --data-binary @scripts/training/sample_api_body.txt
