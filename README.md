# Plagiarism Checker AI Agent

A Mastra-based plagiarism detection agent with Flask JSON-RPC bridge.

## Prerequisites

- **Node.js v20 LTS** (required - Node.js v23 has compatibility issues)
- Python 3.8+
- OpenAI API Key

## Setup

1. Install dependencies:

```bash
cd plagiarism_checker
npm install
pip install Flask Flask-JSONRPC python-dotenv requests
```

2. Set up environment variables:
   Create a `.env` file in the `plagiarism_checker` directory:

```
MASTRA_AGENT_URL=http://127.0.0.1:4111/a2a/plagiarismAgent
OPENAI_API_KEY=your-api-key-here
FLASK_PORT=3000
```

3. Build the Mastra application:

```bash
npm run build
```

4. Start Mastra server:

```bash
npx mastra start
```

5. Start Flask server (in a separate terminal):

```bash
cd plagiarism_checker
python app.py
```

## Testing

Test the plagiarism checker using curl:

**Git Bash / WSL:**

```bash
curl -X POST http://127.0.0.1:3000/api \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"plagiarism.check","params":{"text":"The quick brown fox jumps over the lazy dog.","reference_text":"The quick brown fox jumps over a lazy hound.","threshold":0.75},"id":1}'
```

**PowerShell:**

```powershell
$body = @{
  jsonrpc = "2.0"
  method  = "plagiarism.check"
  params  = @{
    text = "The quick brown fox jumps over the lazy dog."
    reference_text = "The quick brown fox jumps over a lazy hound."
    threshold = 0.75
  }
  id = 1
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:3000/api -Method Post `
  -ContentType "application/json" -Body $body
```

## Node.js Version Issue

If you see an error like `TypeError: Cannot assign to read only property 'name'`, you're using Node.js v23. Switch to Node.js v20 LTS:

1. Install Node.js v20 LTS from https://nodejs.org/
2. Or use nvm to switch versions:
   ```bash
   nvm install 20
   nvm use 20
   ```

## Architecture

- **Mastra Agent**: Handles plagiarism detection using AI and custom tools
- **Flask JSON-RPC Bridge**: Provides a JSON-RPC 2.0 interface for external clients
- **Plagiarism Tool**: Performs text similarity analysis using cosine similarity, Jaccard similarity, and n-gram overlap
