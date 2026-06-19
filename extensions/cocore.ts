import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Constants ────────────────────────────────────────────────────────────────

const BASE_URL = "https://console.cocore.dev/api/v1";
const CONFIG_DIR = join(homedir(), ".pi", "agent");
const CONFIG_PATH = join(CONFIG_DIR, "cocore-config.json");

// ── Model detection ──────────────────────────────────────────────────────────

/**
 * Check if a model is served through the Co/Core provider.
 */
function isCocoreModel(model: { model?: string; provider?: string }): boolean {
  return model.provider === "cocore";
}

/**
 * Check if a model is a Gemma variant (3 or 4) from Co/Core.
 */
function isGemmaModel(model: { model?: string; provider?: string }): boolean {
  return isCocoreModel(model) && /\bgemma\b/i.test(model.model ?? "");
}

/**
 * Check if a model is a Qwen variant (2.5 or 3) from Co/Core.
 */
function isQwenModel(model: { model?: string; provider?: string }): boolean {
  return isCocoreModel(model) && /\bqwen\b/i.test(model.model ?? "");
}

// ── Tool instruction builders ────────────────────────────────────────────────

/** Extract a JSON substring starting at openBraceIdx using brace counting. */
function extractJsonFrom(text: string, openBraceIdx: number): string | null {
  if (text[openBraceIdx] !== "{") return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = openBraceIdx; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth++;
    if (ch === "}") {
      depth--;
      if (depth === 0) {
        return text.slice(openBraceIdx, i + 1);
      }
    }
  }
  return null;
}

interface ToolCallMatch {
  start: number;
  end: number;
  name: string;
  arguments: Record<string, unknown>;
}

/**
 * Build tool-calling instructions for Qwen models.
 * Qwen 2.5/3 use: <tool_call>\n{"name":"func","arguments":{...}}\n</tool_call>
 */
function buildQwenToolInstructions(tools: Array<{ type: string; function: { name: string; description?: string; parameters?: { type: string; properties?: Record<string, { type?: string; description?: string }>; required?: string[] } } }>): string {
  let text = "\n\n# Tools\n\n";
  text += "You have access to the following functions. To call a function, you MUST output ONLY a JSON block in this exact format:\n\n";
  text += "<tool_call>\n";
  text += '{"name": "function_name", "arguments": {"param1": "value1"}}\n';
  text += "</tool_call>\n\n";
  text += "Available functions:\n";
  for (const tool of tools) {
    const func = tool.function;
    text += `\n### ${func.name}\n`;
    text += `${func.description || "No description"}\n`;
    if (func.parameters?.properties) {
      const required = func.parameters.required || [];
      for (const [pname, pdef] of Object.entries(func.parameters.properties)) {
        const req = required.includes(pname) ? " (required)" : "";
        text += `  - ${pname}${req}: ${pdef.description || pdef.type || "any"}\n`;
      }
    }
  }
  text += "\nWhen you need to use a tool, output ONLY the <tool_call> block. Do not include any other text, explanations, or code. The tool result will be provided to you.\n";
  return text;
}

/**
 * Build tool-calling instructions for Gemma models.
 * Gemma 3: <|tool_call|>func{json}<|tool_call|>func2{json}
 * Gemma 4: <|tool_call>call:func{json}<|tool_call|>
 *
 * We use the Gemma 3 format since it's simpler and works for both variants.
 */
function buildGemmaToolInstructions(tools: Array<{ type: string; function: { name: string; description?: string; parameters?: { type: string; properties?: Record<string, { type?: string; description?: string }>; required?: string[] } } }>): string {
  let text = "\n\n# Tools\n\n";
  text += "You have access to the following functions. To call a function, you MUST use this exact format:\n\n";
  text += '<|tool_call|>function_name{"param1": "value1"}\n\n';
  text += "For multiple function calls, separate each with <|tool_call|>:\n\n";
  text += '<|tool_call|>first_func{"param":"value"}<|tool_call|>second_func{"param":"value"}\n\n';
  text += "Available functions:\n";
  for (const tool of tools) {
    const func = tool.function;
    text += `\n### ${func.name}\n`;
    text += `${func.description || "No description"}\n`;
    if (func.parameters?.properties) {
      const required = func.parameters.required || [];
      for (const [pname, pdef] of Object.entries(func.parameters.properties)) {
        const req = required.includes(pname) ? " (required)" : "";
        text += `  - ${pname}${req}: ${pdef.description || pdef.type || "any"}\n`;
      }
    }
  }
  text += "\nWhen you need to use a tool, output ONLY the tool call. Do not include any other text, explanations, or code.\n";
  return text;
}

// ── Tool call parsers ────────────────────────────────────────────────────────

/**
 * Parse Gemma 3 tool calls from text.
 *
 * Gemma 3 format: <|tool_call|>func{json}<|tool_call|>func2{json2}
 * `<|tool_call|>` acts as a delimiter. Each segment between delimiters
 * is a function call: `func_name{json_args}`.
 */
function parseGemma3ToolCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];
  const delim = "<|tool_call|>";
  let searchFrom = 0;

  while (searchFrom < text.length) {
    const delimIdx = text.indexOf(delim, searchFrom);
    if (delimIdx === -1) break;

    // Content starts after the delimiter
    const contentStart = delimIdx + delim.length;

    // Find the next delimiter or end of text
    const nextDelim = text.indexOf(delim, contentStart);
    const contentEnd = nextDelim === -1 ? text.length : nextDelim;

    const segment = text.slice(contentStart, contentEnd).trim();
    if (segment.length > 0) {
      // Parse func_name{json} from segment
      const braceIdx = segment.indexOf("{");
      if (braceIdx > 0) {
        const funcName = segment.slice(0, braceIdx).trim();
        const jsonStr = extractJsonFrom(segment, braceIdx);
        if (jsonStr && funcName) {
          try {
            const args = JSON.parse(jsonStr);
            results.push({
              start: delimIdx,
              end: contentEnd,
              name: funcName,
              arguments: args,
            });
          } catch {
            // Malformed JSON — skip
          }
        }
      }
    }

    searchFrom = nextDelim === -1 ? text.length : nextDelim;
  }

  return results;
}

/**
 * Parse Gemma 4 tool calls from text.
 *
 * Gemma 4 format: <|tool_call>call:func_name{json}<|tool_call|>
 *
 * Note: The opening tag uses `>` while the closing tag uses `|>`.
 */
const GEMMA4_TOOL_CALL_RE = /<\|tool_call>call:([a-zA-Z0-9_]+)\s*(\{.+?)<\|tool_call\|>/g;

function parseGemma4ToolCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];
  let match;
  GEMMA4_TOOL_CALL_RE.lastIndex = 0;
  while ((match = GEMMA4_TOOL_CALL_RE.exec(text)) !== null) {
    const name = match[1];
    const argsStr = match[2];
    // The lazy quantifier may not capture nested braces fully — use extractJson
    const braceIdx = argsStr.indexOf("{");
    const fullJson = braceIdx >= 0 ? extractJsonFrom(argsStr, braceIdx) : argsStr;
    try {
      const args = JSON.parse(fullJson ?? argsStr);
      results.push({
        start: match.index,
        end: GEMMA4_TOOL_CALL_RE.lastIndex,
        name,
        arguments: args,
      });
    } catch {
      // Malformed JSON — skip
    }
  }
  return results;
}

/**
 * Parse Qwen tool calls from text.
 *
 * Qwen format:
 *   1. <tool_call>\n{"name":"func","arguments":{...}}\n</tool_call>
 *   2. Bare {"name":"func","arguments":{...}} (one per line or separated by blank lines)
 *   3. Multiple blocks separated by newlines
 */
function parseQwenToolCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];

  // Strategy 1: Look for <tool_call>...</tool_call> blocks
  const blockRe = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g;
  let match;
  while ((match = blockRe.exec(text)) !== null) {
    const jsonStr = match[1];
    try {
      const parsed = JSON.parse(jsonStr);
      if (parsed && typeof parsed.name === "string" && typeof parsed.arguments === "object") {
        results.push({
          start: match.index,
          end: blockRe.lastIndex,
          name: parsed.name,
          arguments: parsed.arguments,
        });
      }
    } catch {
      // Malformed JSON — skip
    }
  }

  if (results.length > 0) return results;

  // Strategy 2: Find bare {"name":"...","arguments":{...}} objects
  // Search for JSON objects that look like tool calls
  const bareRe = /\{\s*"name"\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*,\s*"arguments"\s*:/g;
  while ((match = bareRe.exec(text)) !== null) {
    const jsonStr = extractJsonFrom(text, match.index);
    if (jsonStr) {
      try {
        const parsed = JSON.parse(jsonStr);
        if (parsed && typeof parsed.name === "string" && typeof parsed.arguments === "object") {
          results.push({
            start: match.index,
            end: match.index + jsonStr.length,
            name: parsed.name,
            arguments: parsed.arguments,
          });
          // Skip past this match for the next search
          bareRe.lastIndex = match.index + jsonStr.length;
        }
      } catch {
        // Malformed JSON — skip
      }
    }
  }

  return results;
}

/**
 * Parse tool calls from text for a given model family.
 */
function parseToolCalls(
  text: string,
  modelFamily: "gemma" | "qwen",
): ToolCallMatch[] {
  if (modelFamily === "gemma") {
    // Try Gemma 4 format first (more specific), fall back to Gemma 3
    const g4 = parseGemma4ToolCalls(text);
    if (g4.length > 0) return g4;
    return parseGemma3ToolCalls(text);
  }
  if (modelFamily === "qwen") {
    return parseQwenToolCalls(text);
  }
  return [];
}

// ── Message fixer ────────────────────────────────────────────────────────────

/**
 * Post-process an assistant message to extract model-native tool calls
 * from text content and convert them to structured toolCall blocks.
 */
function fixCocoreToolCalls(
  message: { role: string; content: Array<{ type: string; text?: string }> },
  modelFamily: "gemma" | "qwen",
): { role: string; content: Array<{ type: string; text?: string }> } {
  if (message.role !== "assistant") return message;

  // Collect all text blocks and their positions
  const textBlocks: Array<{ index: number; text: string }> = [];
  for (let i = 0; i < message.content.length; i++) {
    const block = message.content[i];
    if (block.type === "text" && typeof block.text === "string") {
      textBlocks.push({ index: i, text: block.text });
    }
  }

  if (textBlocks.length === 0) return message;

  // Check all text blocks for tool calls
  let hasToolCalls = false;
  const allMatches: Array<{
    textIndex: number;
    matchStart: number;
    matchEnd: number;
    name: string;
    arguments: Record<string, unknown>;
  }> = [];

  for (const tb of textBlocks) {
    const matches = parseToolCalls(tb.text, modelFamily);
    if (matches.length > 0) {
      hasToolCalls = true;
      for (const m of matches) {
        allMatches.push({
          textIndex: tb.index,
          matchStart: m.start,
          matchEnd: m.end,
          name: m.name,
          arguments: m.arguments,
        });
      }
    }
  }

  if (!hasToolCalls) return message;

  // Build new content: split text blocks around tool-call regions, insert ToolCall blocks
  const newContent: Array<
    { type: string; text?: string } | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }
  > = [];

  for (let i = 0; i < message.content.length; i++) {
    const block = message.content[i];
    const blockMatches = allMatches.filter((m) => m.textIndex === i);

    if (block.type === "text" && typeof block.text === "string" && blockMatches.length > 0) {
      const sorted = [...blockMatches].sort((a, b) => a.matchStart - b.matchStart);

      let lastEnd = 0;
      for (const m of sorted) {
        // Text before this match
        if (m.matchStart > lastEnd) {
          const prefix = block.text.slice(lastEnd, m.matchStart);
          if (prefix.length > 0) {
            newContent.push({ type: "text", text: prefix });
          }
        }
        // Insert tool call block
        const toolCallId = `cocore-tc-${m.name}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        newContent.push({
          type: "toolCall",
          id: toolCallId,
          name: m.name,
          arguments: m.arguments,
        });
        lastEnd = m.matchEnd;
      }

      // Text after the last match
      if (lastEnd < block.text.length) {
        const suffix = block.text.slice(lastEnd);
        if (suffix.length > 0) {
          newContent.push({ type: "text", text: suffix });
        }
      }
    } else {
      newContent.push(block);
    }
  }

  return { ...message, content: newContent };
}

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

// ── Get model family for a cocore model ──────────────────────────────────────

function getModelFamily(modelId: string): "gemma" | "qwen" | null {
  const id = modelId.toLowerCase();
  if (id.includes("gemma")) return "gemma";
  if (id.includes("qwen")) return "qwen";
  return null;
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
    registerEventHandlers(pi);
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
      registerEventHandlers(pi);
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

    registerEventHandlers(pi);
  });

  // Register event handlers immediately (setup command still works)
  registerEventHandlers(pi);
}

/**
 * Register all event handlers. Called both on happy path and after setup.
 */
let handlersRegistered = false;

function registerEventHandlers(pi: ExtensionAPI) {
  if (handlersRegistered) return;
  handlersRegistered = true;

  // ── Inject tool-calling instructions into the prompt ────────────────────
  // Co/Core's OpenAI-compatible API does not translate tool definitions
  // into model-native format for Gemma and Qwen models. We inject text
  // instructions into the system prompt and strip the tools array so the
  // models understand how to call tools.
  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== "cocore") return;

    const modelId = ctx.model.id.toLowerCase();
    const family = getModelFamily(modelId);
    if (!family) return; // Not a Gemma or Qwen model (e.g., Llama)

    const payload = event.payload as Record<string, unknown>;
    const tools = payload.tools as Array<{ type: string; function: { name: string; description?: string; parameters?: { type: string; properties?: Record<string, { type?: string; description?: string }>; required?: string[] } } }> | undefined;
    if (!tools || tools.length === 0) return;

    // Build tool format instructions
    const instructions =
      family === "qwen"
        ? buildQwenToolInstructions(tools)
        : buildGemmaToolInstructions(tools);

    // Append to the system/developer message, or insert one if missing
    const messages = payload.messages as Array<{ role: string; content: string | Array<{ type: string; text?: string }> }> | undefined;
    if (messages && messages.length > 0) {
      let found = false;
      for (const msg of messages) {
        if (msg.role === "system" || msg.role === "developer") {
          // Check if we already injected instructions in a previous turn.
          // Our instructions always contain this unique phrase.
          const existing = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
          if (existing.includes("You have access to the following functions. To call a function, you MUST")) {
            found = true;
            break;
          }
          if (typeof msg.content === "string") {
            msg.content += instructions;
          } else if (Array.isArray(msg.content)) {
            msg.content.push({ type: "text", text: instructions });
          }
          found = true;
          break;
        }
      }
      // If no system/developer message exists, prepend one
      if (!found) {
        messages.unshift({ role: "system", content: instructions });
      }
    } else {
      // No messages array — create one with a system message
      payload.messages = [{ role: "system", content: instructions }];
    }

    // Strip the tools array — model doesn't understand OpenAI tool format
    delete payload.tools;
    // Also strip tool_choice if present
    delete payload.tool_choice;

    return payload;
  });

  // ── Fix tool calls in text content for all cocore models ──────────────
  // Gemma and Qwen models output tool calls as text tokens instead of
  // structured tool_calls. We detect and convert them after streaming.
  pi.on("message_end", (event) => {
    const modelId = (event.message as any).model as string | undefined;
    const provider = (event.message as any).provider as string | undefined;

    if (provider !== "cocore") return;

    const family = getModelFamily(modelId ?? "");
    if (!family) return; // Not a Gemma or Qwen model

    const fixed = fixCocoreToolCalls(event.message as any, family);
    if (fixed === event.message) return;

    // Return the modified message to replace the original
    return { message: fixed as any };
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