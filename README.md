# pi-cocore

Run open-source models on Apple Silicon via [cocore.dev](https://cocore.dev) — directly inside [pi](https://pi.dev).

## Install

```bash
pi install git:github.com/willnewby/pi-cocore
```

## Setup

1. Get your API key from [console.cocore.dev](https://console.cocore.dev)
2. Start pi — on first run you'll be prompted for the key
3. Or run `/cocore-setup` at any time to configure (or change) your key

## Usage

Once configured, Co/Core models appear in the model picker (`Ctrl+P`) alongside your other providers. Supported models include:

| Model | Context |
|-------|---------|
| Qwen 2.5 (0.5B – 32B) | 32K – 128K |
| Gemma 3 / 4 | 32K – 128K |
| Llama 3.3 70B | 128K |

Capabilities (context window, max tokens, reasoning) are automatically derived from each model's ID.

## Requirements

- pi (latest)
- A Co/Core API key from [console.cocore.dev](https://console.cocore.dev)
