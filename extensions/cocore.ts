import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Constants ────────────────────────────────────────────────────────────────

const BASE_URL = "https://console.cocore.dev/api/v1";
const CONFIG_DIR = join(homedir(), ".pi", "agent");
const CONFIG_PATH = join(CONFIG_DIR, "cocore-config.json");

// ── Types ────────────────────────────────────────────────────────────────────

interface CocoreConfig {
  apiKey: string;
}

interface CocoreModelEntry {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
}

interface CocoreModelsResponse {
  object: string;
  data: CocoreModelEntry[];
}

interface ModelCapabilities {
  reasoning: boolean;
  input: ["text"];
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  contextWindow: number;
  maxTokens: number;
}

// ── Config persistence ───────────────────────────────────────────────────────

function loadConfig(): CocoreConfig | null {
  try {
    if (existsSync(CONFIG_PATH)) {
      const raw = readFileSync(CONFIG_PATH, "utf-8");
      return JSON.parse(raw) as CocoreConfig;
    }
  } catch {
    // Corrupt or missing — treat as unconfigured
  }
  return null;
}

function saveConfig(config: CocoreConfig): void {
  mkdirSync(CONFIG_DIR, { recursive: true });
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), "utf-8");
}

// ── Model capability derivation ──────────────────────────────────────────────

/**
 * Derive model capabilities from the model ID since the /models endpoint
 * only returns minimal metadata (id, object, created, owned_by).
 *
 * Recognized families fall back to sensible defaults for unknown models.
 */
function deriveModelCapabilities(modelId: string): ModelCapabilities {
  const id = modelId.toLowerCase();

  // Default: conservative values for a modern small-to-mid LLM
  let contextWindow = 32_768;
  let maxTokens = 8_192;

  // ── Qwen 2.5 family ──────────────────────────────────────────────────
  if (id.includes("qwen2.5")) {
    // Smaller Qwen2.5 variants (0.5B, 1.5B, 3B): 32K context
    if (id.includes("0.5b") || id.includes("1.5b") || id.includes("3b")) {
      contextWindow = 32_768;
    } else {
      // 7B, 14B, 32B, 72B: 128K context
      contextWindow = 128_000;
    }
    maxTokens = 8_192;
  }

  // ── Qwen 3 family ────────────────────────────────────────────────────
  else if (id.includes("qwen3")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Gemma family ─────────────────────────────────────────────────────
  else if (id.includes("gemma-3")) {
    contextWindow = 32_768;
    maxTokens = 8_192;
  } else if (id.includes("gemma-4")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Llama 3 family ───────────────────────────────────────────────────
  else if (
    id.includes("llama-3.3") ||
    id.includes("llama-3.2") ||
    id.includes("llama-3.1")
  ) {
    contextWindow = 128_000;
    maxTokens = 16_384;
  } else if (id.includes("llama-4")) {
    contextWindow = 128_000;
    maxTokens = 16_384;
  } else if (id.includes("llama")) {
    // Catch-all for future Llama releases
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Mistral / Mixtral ────────────────────────────────────────────────
  else if (id.includes("mistral") || id.includes("mixtral")) {
    contextWindow = 32_768;
    maxTokens = 8_192;
    if (id.includes("large") || id.includes("8x")) {
      contextWindow = 128_000;
    }
  }

  // ── DeepSeek ─────────────────────────────────────────────────────────
  else if (id.includes("deepseek")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
    if (id.includes("r1")) {
      // R1 is a reasoning model
      return {
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow,
        maxTokens,
      };
    }
  }

  // ── Phi ──────────────────────────────────────────────────────────────
  else if (id.includes("phi-4") || id.includes("phi-3")) {
    contextWindow = 128_000;
    maxTokens = 4_096;
    if (id.includes("vision") || id.includes("multimodal")) {
      contextWindow = 128_000;
      maxTokens = 4_096;
    }
  }

  return {
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens,
  };
}

// ── Provider registration ────────────────────────────────────────────────────

async function registerCocoreProvider(pi: ExtensionAPI, apiKey: string): Promise<number> {
  const response = await fetch(`${BASE_URL}/models`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(
      `Co/Core /models returned ${response.status} ${response.statusText}: ${body.slice(0, 200)}`
    );
  }

  const payload = (await response.json()) as CocoreModelsResponse;

  if (payload.object !== "list" || !Array.isArray(payload.data)) {
    throw new Error("[cocore] Unexpected /models response format");
  }

  const models = payload.data
    .filter((m) => m.id !== "stub") // Skip internal stub entry
    .map((m) => {
      const caps = deriveModelCapabilities(m.id);
      return {
        id: m.id,
        name: m.id.split("/").pop() ?? m.id,
        ...caps,
      };
    });

  pi.registerProvider("cocore", {
    name: "Co/Core",
    baseUrl: BASE_URL,
    apiKey,
    api: "openai-completions",
    models,
  });

  return models.length;
}

// ── Extension entry point ────────────────────────────────────────────────────

export default async function (pi: ExtensionAPI) {
  const config = loadConfig();

  // ── Happy path: API key already saved ──────────────────────────────────
  if (config?.apiKey) {
    try {
      const count = await registerCocoreProvider(pi, config.apiKey);
      console.log(`[cocore] Registered ${count} model(s)`);
    } catch (err) {
      console.error(`[cocore] ${err instanceof Error ? err.message : err}`);
    }
    return;
  }

  // ── First run: prompt for API key on session start ─────────────────────
  let setupDone = false;

  pi.on("session_start", async (_event, ctx) => {
    if (setupDone) return;
    setupDone = true;

    // Re-check config in case another process saved it
    const fresh = loadConfig();
    if (fresh?.apiKey) {
      try {
        await registerCocoreProvider(pi, fresh.apiKey);
        ctx.ui.notify("Co/Core provider registered!", "info");
      } catch (err) {
        ctx.ui.notify(
          `Co/Core registration failed: ${err instanceof Error ? err.message : err}`,
          "error"
        );
      }
      return;
    }

    const apiKey = await ctx.ui.input(
      "Enter your Co/Core API key (from console.cocore.dev):",
      { password: true }
    );

    if (!apiKey?.trim()) {
      ctx.ui.notify(
        "Co/Core setup skipped. Run /cocore-setup when ready.",
        "warning"
      );
      return;
    }

    saveConfig({ apiKey: apiKey.trim() });

    try {
      const count = await registerCocoreProvider(pi, apiKey.trim());
      ctx.ui.notify(`Co/Core ready — ${count} model(s) available.`, "info");
      console.log(`[cocore] Registered ${count} model(s)`);
    } catch (err) {
      ctx.ui.notify(
        `Co/Core: ${err instanceof Error ? err.message : err}`,
        "error"
      );
    }
  });

  // ── Manual setup command ───────────────────────────────────────────────
  pi.registerCommand("cocore-setup", {
    description: "Configure or reconfigure your Co/Core API key",
    handler: async (_args, ctx) => {
      const existing = loadConfig();
      const prompt = existing?.apiKey
        ? "Enter a new Co/Core API key (leave blank to keep current):"
        : "Enter your Co/Core API key (from console.cocore.dev):";

      const apiKey = await ctx.ui.input(prompt, { password: true });

      if (!apiKey?.trim()) {
        if (existing?.apiKey) {
          ctx.ui.notify("Co/Core API key unchanged.", "info");
        } else {
          ctx.ui.notify("Co/Core setup skipped.", "warning");
        }
        return;
      }

      saveConfig({ apiKey: apiKey.trim() });
      ctx.ui.notify(
        "Co/Core API key saved. Restart pi or run /reload to activate.",
        "info"
      );
    },
  });
}