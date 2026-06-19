import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  createAssistantMessageEventStream,
  calculateCost,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  type Api,
  type AssistantMessage,
  type StopReason,
  type ToolCall,
  type Message,
  type Tool,
  type Usage,
} from "@earendil-works/pi-ai";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Constants ────────────────────────────────────────────────────────────────

const BASE_URL = "https://console.cocore.dev/api/v1";
const CONFIG_DIR = join(homedir(), ".pi", "agent");
const CONFIG_PATH = join(CONFIG_DIR, "cocore-config.json");

// ── Retry configuration ──────────────────────────────────────────────────────

/** Maximum number of retry attempts for failed requests. */
const MAX_RETRIES = 3;

/** Base delay in ms for exponential backoff. */
const RETRY_BASE_DELAY_MS = 1000;

/** HTTP status codes that should be retried. */
const RETRYABLE_STATUS_CODES = new Set([
  408, // Request Timeout
  425, // Too Early
  429, // Too Many Requests
  500, // Internal Server Error
  502, // Bad Gateway
  503, // Service Unavailable
  504, // Gateway Timeout
]);

/**
 * Check whether an error should trigger a retry.
 */
function isRetryableError(status: number, errorMessage?: string): boolean {
  if (RETRYABLE_STATUS_CODES.has(status)) return true;
  if (errorMessage) {
    const lower = errorMessage.toLowerCase();
    if (lower.includes("idle-timeout") || lower.includes("idle_timeout")) return true;
    if (lower.includes("timeout") || lower.includes("timed out")) return true;
    if (lower.includes("rate limit") || lower.includes("too many requests")) return true;
  }
  return false;
}

/**
 * Sleep for `ms` milliseconds.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Calculate the backoff delay for a given retry attempt.
 * Uses exponential backoff with jitter: base * 2^attempt + random jitter.
 */
function backoffDelay(attempt: number): number {
  const exp = Math.min(attempt, 10); // Cap exponent to prevent overflow
  const base = RETRY_BASE_DELAY_MS * Math.pow(2, exp);
  const jitter = Math.random() * base * 0.5;
  return base + jitter;
}

// ── Custom streaming with retry ─────────────────────────────────────────────

/**
 * Build tool instructions for Gemma or Qwen models and inject them
 * into the context's system prompt. Returns a copy of the context with
 * modified system prompt if tools are present and model family matches.
 */
function injectToolInstructions(
  context: Context,
  modelFamily: "gemma" | "qwen",
): Context {
  if (!context.tools || context.tools.length === 0) return context;

  const piTools = context.tools as Tool<any>[];
  // Convert Tool objects to the format expected by instruction builders
  const toolDescs = piTools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters as any,
    },
  }));

  const instructions =
    modelFamily === "qwen"
      ? buildQwenToolInstructions(toolDescs)
      : buildGemmaToolInstructions(toolDescs);

  // Check if instructions are already injected
  const existingCheck = "You have access to the following functions. To call a function, you MUST";
  if (context.systemPrompt && context.systemPrompt.includes(existingCheck)) {
    return context;
  }

  return {
    ...context,
    systemPrompt: (context.systemPrompt ?? "") + instructions,
    // Don't send tools in OpenAI format — the model uses text instructions
    tools: undefined,
  };
}

/**
 * Convert pi's internal message format to OpenAI Chat Completions format.
 */
function convertMessagesForOpenAI(messages: Message[]): unknown[] {
  const result: unknown[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
      } else {
        const parts = msg.content.map((c) => {
          if (c.type === "text") return { type: "text", text: c.text };
          return {
            type: "image_url",
            image_url: { url: `data:${c.mimeType};base64,${c.data}` },
          };
        });
        result.push({ role: "user", content: parts });
      }
    } else if (msg.role === "assistant") {
      const content: unknown[] = [];
      const toolCalls: unknown[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          content.push({ type: "text", text: block.text });
        } else if (block.type === "thinking") {
          // Convert thinking to text for OpenAI compat
          content.push({ type: "text", text: `<thinking>${block.thinking}</thinking>` });
        } else if (block.type === "toolCall") {
          toolCalls.push({
            id: block.id,
            type: "function",
            function: {
              name: block.name,
              arguments: JSON.stringify(block.arguments),
            },
          });
        }
      }
      const assistantMsg: Record<string, unknown> = {
        role: "assistant",
        content: content.length > 0 ? content : null,
      };
      if (toolCalls.length > 0) {
        assistantMsg.tool_calls = toolCalls;
      }
      result.push(assistantMsg);
    } else if (msg.role === "toolResult") {
      const content = msg.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      result.push({
        role: "tool",
        tool_call_id: msg.toolCallId,
        content,
      });
    }
  }

  return result;
}

/**
 * Custom streaming implementation for Co/Core provider with retry logic.
 *
 * Handlers:
 * - Injects tool-calling instructions for Gemma/Qwen models
 * - Retries on idle-timeout and transient server errors
 * - Parses OpenAI-compatible SSE stream
 */
function streamCocore(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    const maxRetries = options?.maxRetries ?? MAX_RETRIES;
    const signal = options?.signal;
    const apiKey = options?.apiKey;

    // Detect model family for tool instruction injection
    const family = getModelFamily(model.id);
    const effectiveContext = family
      ? injectToolInstructions(context, family)
      : context;

    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    let lastErrorMessage: string | undefined;
    let textContentIndex: number | null;
    const toolCallAccumulators: Map<
      number,
      { id: string; name: string; json: string; contentIdx: number }
    > = new Map();

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (signal?.aborted) {
        output.stopReason = "aborted";
        output.errorMessage = "Request was aborted";
        stream.push({ type: "error", reason: "aborted", error: output });
        stream.end();
        return;
      }

      if (attempt > 0) {
        const delay = backoffDelay(attempt - 1);
        console.log(
          `[cocore] Retry attempt ${attempt}/${maxRetries} after ${Math.round(delay)}ms (previous error: ${lastErrorMessage})`,
        );
        await sleep(delay);

        // Re-check abort after waiting
        if (signal?.aborted) {
          output.stopReason = "aborted";
          output.errorMessage = "Request was aborted";
          stream.push({ type: "error", reason: "aborted", error: output });
          stream.end();
          return;
        }
      }

      try {
        // Reset output content for this attempt (in case of retry)
        output.content = [];
        output.usage = {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        };
        output.stopReason = "stop";

        // Build request payload
        const messages = convertMessagesForOpenAI(effectiveContext.messages);
        if (effectiveContext.systemPrompt) {
          messages.unshift({
            role: "system",
            content: effectiveContext.systemPrompt,
          });
        }

        const body: Record<string, unknown> = {
          model: model.id,
          messages,
          stream: true,
          stream_options: { include_usage: true },
        };

        if (options?.maxTokens) {
          body.max_tokens = options.maxTokens;
        }
        if (options?.temperature !== undefined) {
          body.temperature = options.temperature;
        }

        // Only include tools in OpenAI format for non-Gemma/Qwen models
        if (!family && effectiveContext.tools && effectiveContext.tools.length > 0) {
          body.tools = effectiveContext.tools.map((t: Tool<any>) => ({
            type: "function",
            function: {
              name: t.name,
              description: t.description,
              parameters: t.parameters,
            },
          }));
        }

        console.log(`[cocore] sending request (attempt ${attempt + 1}/${maxRetries + 1})`);

        const headers: Record<string, string> = {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        };

        // Merge model-level and provider-level headers
        if (model.headers) Object.assign(headers, model.headers);
        if (options?.headers) Object.assign(headers, options.headers);

        const fetchTimeout = options?.timeoutMs ?? 600_000; // Default 10 min
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), fetchTimeout);

        // Link to parent abort signal if provided
        const onAbort = () => controller.abort();
        signal?.addEventListener("abort", onAbort, { once: true });

        let response: Response;
        try {
          response = await fetch(`${model.baseUrl || BASE_URL}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify(body),
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timeoutId);
          signal?.removeEventListener("abort", onAbort);
        }

        if (!response.ok) {
          const errorText = await response.text().catch(() => "");
          lastErrorMessage = errorText.slice(0, 500);

          if (isRetryableError(response.status, errorText) && attempt < maxRetries) {
            console.log(
              `[cocore] Request failed with status ${response.status}: ${errorText.slice(0, 200)}. Will retry.`,
            );
            continue; // Retry
          }

          // Not retryable, or exhausted retries
          output.stopReason = "error";
          output.errorMessage =
            `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}: ${errorText.slice(0, 300)}`;
          stream.push({ type: "error", reason: "error", error: output });
          stream.end();
          return;
        }

        // Success — parse the SSE stream
        stream.push({ type: "start", partial: output });

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("Response body is not readable");
        }

        const decoder = new TextDecoder();
        let buffer = "";
        textContentIndex = null;
        toolCallAccumulators.clear();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE lines
            const lines = buffer.split("\n");
            // Keep the last (potentially incomplete) line in the buffer
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6).trim();
              if (data === "[DONE]") continue;

              let chunk: Record<string, unknown>;
              try {
                chunk = JSON.parse(data);
              } catch {
                continue;
              }

              const choices = chunk.choices as
                | Array<Record<string, unknown>>
                | undefined;
              if (!choices || choices.length === 0) continue;
              const choice = choices[0];
              const delta = choice.delta as Record<string, unknown> | undefined;

              // Handle usage info
              if (chunk.usage) {
                const u = chunk.usage as Record<string, unknown>;
                output.usage.input = (u.prompt_tokens as number) ?? 0;
                output.usage.output = (u.completion_tokens as number) ?? 0;
                const details = u.prompt_tokens_details as Record<string, number> | undefined;
                output.usage.cacheRead = details?.cached_tokens ?? 0;
                output.usage.totalTokens =
                  output.usage.input +
                  output.usage.output +
                  output.usage.cacheRead +
                  output.usage.cacheWrite;
                calculateCost(model, output.usage as Usage);
              }

              // Handle finish reason
              if (choice.finish_reason) {
                const reason = choice.finish_reason as string;
                if (reason === "tool_calls") {
                  output.stopReason = "toolUse";
                } else if (reason === "length") {
                  output.stopReason = "length";
                } else if (reason === "stop") {
                  output.stopReason = "stop";
                }
              }

              if (!delta || Object.keys(delta as Record<string, unknown>).length === 0) continue;

              // Text content
              if (delta.content) {
                if (textContentIndex === null) {
                  textContentIndex = output.content.length;
                  output.content.push({ type: "text", text: "" });
                  stream.push({
                    type: "text_start",
                    contentIndex: textContentIndex,
                    partial: output,
                  });
                }
                const block = output.content[textContentIndex];
                if (block.type === "text") {
                  block.text += delta.content as string;
                  stream.push({
                    type: "text_delta",
                    contentIndex: textContentIndex,
                    delta: delta.content as string,
                    partial: output,
                  });
                }
              }

              // Tool calls
              const toolCalls = delta.tool_calls as
                | Array<Record<string, unknown>>
                | undefined;
              if (toolCalls) {
                for (const tc of toolCalls) {
                  const idx = tc.index as number;

                  let accum = toolCallAccumulators.get(idx);
                  if (!accum) {
                    const contentIdx = output.content.length;
                    accum = {
                      id: (tc.id as string) || "",
                      name: "",
                      json: "",
                      contentIdx,
                    };
                    toolCallAccumulators.set(idx, accum);

                    // Add placeholder to output content
                    output.content.push({
                      type: "toolCall",
                      id: accum.id,
                      name: "",
                      arguments: {},
                    });
                    stream.push({
                      type: "toolcall_start",
                      contentIndex: contentIdx,
                      partial: output,
                    });
                  }

                  if (tc.id) accum.id = tc.id as string;

                  const fn = tc.function as Record<string, unknown> | undefined;
                  if (fn) {
                    if (fn.name) accum.name = fn.name as string;
                    if (fn.arguments) accum.json += fn.arguments as string;
                  }

                  const block = output.content[accum.contentIdx];
                  if (block && block.type === "toolCall") {
                    block.id = accum.id;
                    block.name = accum.name;
                    try {
                      block.arguments = JSON.parse(accum.json);
                    } catch {
                      // Partial JSON — keep previous parse result
                    }
                    stream.push({
                      type: "toolcall_delta",
                      contentIndex: accum.contentIdx,
                      delta: (fn?.arguments as string) ?? "",
                      partial: output,
                    });
                  }
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }

        // End text block if one was started
        if (textContentIndex !== null) {
          const block = output.content[textContentIndex];
          if (block && block.type === "text") {
            stream.push({
              type: "text_end",
              contentIndex: textContentIndex,
              content: block.text,
              partial: output,
            });
          }
        }

        // End any tool call blocks
        for (const [_idx, accum] of toolCallAccumulators) {
          const block = output.content[accum.contentIdx];
          if (block && block.type === "toolCall") {
            try {
              block.arguments = JSON.parse(accum.json);
            } catch {
              // Keep whatever was parsed
            }
            stream.push({
              type: "toolcall_end",
              contentIndex: accum.contentIdx,
              toolCall: block as ToolCall,
              partial: output,
            });
          }
        }

        // Success!
        const hasContent =
          output.content.length > 0 ||
          output.usage.totalTokens > 0;

        if (!hasContent && attempt < maxRetries) {
          // Empty response — likely idle-timeout on the server
          lastErrorMessage = "empty response (likely idle-timeout)";
          console.log(
            `[cocore] Empty response received (0 content, 0 tokens). Will retry.`,
          );
          continue; // Retry
        }

        stream.push({
          type: "done",
          reason: output.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">,
          message: output,
        });
        stream.end();
        return;
      } catch (err) {
        lastErrorMessage = err instanceof Error ? err.message : String(err);

        if (signal?.aborted) {
          output.stopReason = "aborted";
          output.errorMessage = "Request was aborted";
          stream.push({ type: "error", reason: "aborted", error: output });
          stream.end();
          return;
        }

        if (attempt < maxRetries) {
          console.log(
            `[cocore] Request error: ${lastErrorMessage}. Will retry.`,
          );
          continue; // Retry on network errors
        }

        // Exhausted retries
        output.stopReason = "error";
        output.errorMessage = lastErrorMessage;
        stream.push({ type: "error", reason: "error", error: output });
        stream.end();
        return;
      }
    }

    // Should not reach here, but handle it
    output.stopReason = "error";
    output.errorMessage = lastErrorMessage ?? "Unknown error after retries";
    stream.push({ type: "error", reason: "error", error: output });
    stream.end();
  })();

  return stream;
}

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
    streamSimple: streamCocore,
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

  // ── Tool-calling instructions are now injected in the custom stream ───
  // The streamCocore function handles Gemma/Qwen tool instruction injection
  // and strips the tools array before sending to the API.

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