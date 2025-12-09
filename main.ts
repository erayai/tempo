/**
 * Tempo API Proxy - Deno 单文件版本
 * 将 Tempo.build AI 接口封装成 OpenAI 兼容格式
 * 
 * 运行: deno run --allow-net --allow-env main.ts
 */

// ============== 配置 ==============

const CONFIG = {
  clientToken: Deno.env.get("TEMPO_CLIENT_TOKEN") || "",
  canvasId: Deno.env.get("TEMPO_CANVAS_ID") || "",
  port: parseInt(Deno.env.get("PORT") || "3000"),
  // Retry configuration
  maxRetries: parseInt(Deno.env.get("MAX_RETRIES") || "3"),
  retryBaseDelay: 1000,  // 1 second
  retryMaxDelay: 10000,  // 10 seconds
};

if (!CONFIG.clientToken || !CONFIG.canvasId) {
  console.error("❌ 请设置环境变量 TEMPO_CLIENT_TOKEN 和 TEMPO_CANVAS_ID");
  
}

// ============== 导入 ==============

import { parseClientToken, getSessionId, clearSessionCache } from "./session.ts";
import { getCanvasIdFromRequest, validateCanvasId } from "./canvas.ts";

// Re-export canvas functions for backwards compatibility
export { getCanvasIdFromRequest, validateCanvasId } from "./canvas.ts";

// Import and re-export retry functions
import { calculateBackoff, shouldRetry, fetchWithRetry } from "./retry.ts";
export { calculateBackoff, shouldRetry, fetchWithRetry };

// Import rate limiter
import { getRateLimiter, getClientIp } from "./ratelimit.ts";
export { getRateLimiter, getClientIp };

// Import request queue
import { getRequestQueue, QueueFullError } from "./queue.ts";
export { getRequestQueue, QueueFullError };

// Import stats collector
import { getStatsCollector } from "./stats.ts";
export { getStatsCollector };

// Import enhanced logging
import { estimateTokens, sanitizeLog, logRequest, type RequestLogData } from "./logging.ts";
export { estimateTokens, sanitizeLog, logRequest };

// Import auth module
import { extractApiKey, validateApiKey, validateRequest, isAuthEnabled, getCachedAuthConfig } from "./auth.ts";
export { extractApiKey, validateApiKey, validateRequest, isAuthEnabled };

// ============== Token 管理 ==============

let authToken = "";
let tokenExpiry = 0;

/**
 * 解析 JWT Token 的过期时间
 */
function parseTokenExpiry(token: string): number {
  try {
    const jwt = token.replace(/^Bearer\s+/i, "");
    const payload = JSON.parse(atob(jwt.split(".")[1]));
    return (payload.exp || 0) * 1000;
  } catch {
    return 0;
  }
}

// Re-export for backwards compatibility
export { parseClientToken };

async function refreshToken(): Promise<void> {
  console.log("[Tempo] 刷新 Token...");
  
  // 动态获取 Session ID
  const sessionId = await getSessionId(CONFIG.clientToken);
  console.log(`[Tempo] 使用 Session ID: ${sessionId.substring(0, 15)}...`);
  
  const res = await fetch(
    `https://clerk.tempo.build/v1/client/sessions/${sessionId}/tokens?_clerk_js_version=5.56.0-snapshot.v20250530185653`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "Cookie": `__client=${CONFIG.clientToken}`,
        "Origin": "https://app.tempo.build",
      },
    }
  );

  if (!res.ok) {
    // 如果 Token 刷新失败，可能是 Session 过期，清除缓存重试
    if (res.status === 401 || res.status === 403) {
      console.log("[Tempo] Session 可能已过期，清除缓存...");
      clearSessionCache();
    }
    throw new Error(`Token 刷新失败: ${res.status}`);
  }
  
  const data = await res.json();
  if (!data.jwt) throw new Error("响应中没有 JWT");

  authToken = `Bearer ${data.jwt}`;
  tokenExpiry = parseTokenExpiry(authToken);
  console.log("[Tempo] Token 刷新成功，过期时间:", new Date(tokenExpiry).toISOString());
}

async function getValidToken(): Promise<string> {
  if (Date.now() >= tokenExpiry - 10000) {
    await refreshToken();
  }
  return authToken;
}

// ============== 模型列表 ==============

const BASE_MODELS = [
  { id: "claude-4-5-opus", owned_by: "anthropic" },
  { id: "claude-4-5-sonnet", owned_by: "anthropic" },
  { id: "claude-4-5-haiku", owned_by: "anthropic" },
  { id: "claude-4-sonnet", owned_by: "anthropic" },
  { id: "gemini-3-pro", owned_by: "google" },
  { id: "gemini-2.5-pro", owned_by: "google" },
  { id: "gpt-5.1", owned_by: "openai" },
  { id: "auto", owned_by: "tempo" },
];

function getModels() {
  const models = [];
  for (const base of BASE_MODELS) {
    models.push({ id: base.id, object: "model", created: 1700000000, owned_by: base.owned_by });
    models.push({ id: `${base.id}-reasoning`, object: "model", created: 1700000000, owned_by: base.owned_by });
    models.push({ id: `${base.id}-search`, object: "model", created: 1700000000, owned_by: base.owned_by });
    models.push({ id: `${base.id}-reasoning-search`, object: "model", created: 1700000000, owned_by: base.owned_by });
  }
  return { object: "list", data: models };
}

// ============== 请求转换 ==============

function parseModelName(model: string) {
  let baseModel = model;
  let reasoning = false;
  let search = false;

  if (baseModel.endsWith("-reasoning-search") || baseModel.endsWith("-search-reasoning")) {
    baseModel = baseModel.replace(/-reasoning-search$|-search-reasoning$/, "");
    reasoning = true;
    search = true;
  } else {
    if (baseModel.endsWith("-reasoning")) {
      baseModel = baseModel.slice(0, -10);
      reasoning = true;
    }
    if (baseModel.endsWith("-search")) {
      baseModel = baseModel.slice(0, -7);
      search = true;
    }
  }

  return { baseModel, reasoning, search };
}

interface Message {
  role: string;
  content: string;
}

function transformRequest(model: string, messages: Message[]) {
  const { baseModel, reasoning, search } = parseModelName(model);
  
  // 最后一条用户消息作为 user_prompt，其余作为 chat_history
  let userPrompt = "";
  const chatHistory: Message[] = [];
  
  let lastUserIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  for (let i = 0; i < messages.length; i++) {
    if (i === lastUserIndex) {
      userPrompt = messages[i].content;
    } else {
      chatHistory.push({ role: messages[i].role, content: messages[i].content });
    }
  }

  return {
    chat_history: chatHistory,
    chat_history_to_not_summarize: [],
    user_prompt: userPrompt,
    posthog_flags: [],
    all_edited_files: [],
    feature_title: "",
    feature_prd: "",
    mermaid_diagram: "",
    user_selection_context: "",
    selected_elements: "",
    storyboards_in_canvas: "",
    token_soft_limit: 16000,
    layout_mode: "default",
    left_tab: "chat",
    diff_code_snippet: "",
    selected_code_snippet: "",
    use_text_editor: false,
    blind_edited_files: [],
    chat_only_mode: true,
    use_reasoning: reasoning,
    use_search: search,
    is_free_prompt: true,
    code_generation_model: baseModel,
    visual_context: [],
  };
}

// ============== 响应解析 ==============

function parseTempoChunk(line: string): { chunk: string } | null {
  try {
    const outer = JSON.parse(line);
    if (outer.status === -1 && outer.data) {
      const inner = JSON.parse(outer.data);
      if (inner.type === "text-delta" && inner.chunk) {
        return { chunk: inner.chunk };
      }
    }
  } catch { /* ignore */ }
  return null;
}

function createOpenAIDelta(content: string, model: string, finish = false) {
  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      delta: finish ? {} : { content },
      finish_reason: finish ? "stop" : null,
    }],
  };
}

function createOpenAICompletion(content: string, model: string, promptTokens: number) {
  const completionTokens = Math.ceil(content.length / 4);
  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content },
      finish_reason: "stop",
    }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  };
}

// ============== HTTP 处理 ==============

async function handleChat(req: Request): Promise<Response> {
  const startTime = Date.now();
  const body = await req.json();
  const { model = "auto", messages = [], stream = false } = body;
  const { baseModel, reasoning, search } = parseModelName(model);
  
  // Calculate input tokens for logging
  const inputTokens = messages.reduce((acc: number, m: Message) => acc + estimateTokens(m.content), 0);

  if (!messages.length) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    logRequest({
      method: "POST",
      path: "/v1/chat/completions",
      model: baseModel,
      reasoning,
      search,
      duration,
      status: 400,
      success: false,
      error: "messages is required",
    });
    return jsonResponse({ error: { message: "messages is required", type: "invalid_request_error" } }, 400);
  }

  // 获取并验证 Canvas ID
  const canvasId = getCanvasIdFromRequest(req, CONFIG.canvasId);
  if (!validateCanvasId(canvasId)) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    logRequest({
      method: "POST",
      path: "/v1/chat/completions",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens,
      status: 400,
      success: false,
      error: "Invalid canvas_id format",
    });
    return jsonResponse({ error: { message: "Invalid canvas_id format", type: "invalid_request_error" } }, 400);
  }

  const tempoReq = transformRequest(model, messages);
  const token = await getValidToken();

  let tempoRes: Response;
  try {
    tempoRes = await fetchWithRetry(
      `https://api.tempo.build/canvases/${canvasId}/ai/vercel-ai/aiChatMaxAgent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": token,
        },
        body: JSON.stringify(tempoReq),
      },
      CONFIG.maxRetries,
      CONFIG.retryBaseDelay,
      CONFIG.retryMaxDelay
    );
  } catch (error) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    const message = error instanceof Error ? error.message : "Upstream error";
    logRequest({
      method: "POST",
      path: "/v1/chat/completions",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens,
      status: 502,
      success: false,
      error: message,
    });
    return jsonResponse({ 
      error: { 
        message, 
        type: "api_error",
        retryCount: CONFIG.maxRetries
      } 
    }, 502);
  }

  if (!tempoRes.ok) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    logRequest({
      method: "POST",
      path: "/v1/chat/completions",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens,
      status: 502,
      success: false,
      error: "Upstream error",
    });
    return jsonResponse({ error: { message: "Upstream error", type: "api_error" } }, 502);
  }

  if (stream) {
    // 流式响应
    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        const reader = tempoRes.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;
              const delta = parseTempoChunk(line.trim());
              if (delta) {
                const chunk = createOpenAIDelta(delta.chunk, baseModel);
                controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
              }
            }
          }

          // 发送结束标记
          const finalChunk = createOpenAIDelta("", baseModel, true);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(finalChunk)}\n\n`));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          
          // Record successful streaming request
          const duration = Date.now() - startTime;
          getStatsCollector().recordRequest(baseModel, duration, true);
          
          // Enhanced logging for streaming response
          logRequest({
            method: "POST",
            path: "/v1/chat/completions",
            model: baseModel,
            reasoning,
            search,
            canvasId,
            duration,
            inputTokens,
            status: 200,
            success: true,
          });
        } finally {
          controller.close();
          reader.releaseLock();
        }
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } else {
    // 非流式响应
    const reader = tempoRes.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const delta = parseTempoChunk(line.trim());
        if (delta) fullText += delta.chunk;
      }
    }

    // Record successful non-streaming request
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, true);

    const promptTokens = inputTokens;
    const outputTokens = estimateTokens(fullText);
    
    // Enhanced logging for non-streaming response
    logRequest({
      method: "POST",
      path: "/v1/chat/completions",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens: promptTokens,
      outputTokens,
      status: 200,
      success: true,
    });
    
    return jsonResponse(createOpenAICompletion(fullText, baseModel, promptTokens));
  }
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  });
}

// ============== Anthropic API 兼容 (Claude Code) ==============

interface AnthropicMessage {
  role: string;
  content: string | Array<{ type: string; text?: string }>;
}

function normalizeAnthropicContent(content: string | Array<{ type: string; text?: string }>): string {
  if (typeof content === "string") return content;
  return content.map(c => c.text || "").join("");
}

async function handleAnthropicMessages(req: Request): Promise<Response> {
  const startTime = Date.now();
  const body = await req.json();
  const { model = "claude-4-5-sonnet", messages = [], system, stream = false } = body;
  const { baseModel, reasoning, search } = parseModelName(model);

  // 转换 Anthropic 格式到内部格式
  const internalMessages: Message[] = [];
  
  // 添加 system 消息
  if (system) {
    internalMessages.push({ role: "system", content: system });
  }

  // 转换消息
  for (const msg of messages as AnthropicMessage[]) {
    internalMessages.push({
      role: msg.role === "assistant" ? "assistant" : "user",
      content: normalizeAnthropicContent(msg.content),
    });
  }

  // Calculate input tokens for logging
  const inputTokensCount = internalMessages.reduce((acc, m) => acc + estimateTokens(m.content), 0);

  if (!internalMessages.length) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    logRequest({
      method: "POST",
      path: "/v1/messages",
      model: baseModel,
      reasoning,
      search,
      duration,
      status: 400,
      success: false,
      error: "messages is required",
    });
    return jsonResponse({ type: "error", error: { type: "invalid_request_error", message: "messages is required" } }, 400);
  }

  // 获取并验证 Canvas ID
  const canvasId = getCanvasIdFromRequest(req, CONFIG.canvasId);
  if (!validateCanvasId(canvasId)) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    logRequest({
      method: "POST",
      path: "/v1/messages",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens: inputTokensCount,
      status: 400,
      success: false,
      error: "Invalid canvas_id format",
    });
    return jsonResponse({ type: "error", error: { type: "invalid_request_error", message: "Invalid canvas_id format" } }, 400);
  }

  // 直接使用传入的模型名，不做映射
  // 在客户端设置模型 ID 为 Tempo 支持的: claude-4-5-opus, claude-4-5-sonnet 等
  const mappedModel = model;

  const tempoReq = transformRequest(mappedModel, internalMessages);
  const token = await getValidToken();

  let tempoRes: Response;
  try {
    tempoRes = await fetchWithRetry(
      `https://api.tempo.build/canvases/${canvasId}/ai/vercel-ai/aiChatMaxAgent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": token,
        },
        body: JSON.stringify(tempoReq),
      },
      CONFIG.maxRetries,
      CONFIG.retryBaseDelay,
      CONFIG.retryMaxDelay
    );
  } catch (error) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    const message = error instanceof Error ? error.message : "Upstream error";
    logRequest({
      method: "POST",
      path: "/v1/messages",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens: inputTokensCount,
      status: 502,
      success: false,
      error: message,
    });
    return jsonResponse({ 
      type: "error", 
      error: { 
        type: "api_error", 
        message,
        retryCount: CONFIG.maxRetries
      } 
    }, 502);
  }

  if (!tempoRes.ok) {
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, false);
    logRequest({
      method: "POST",
      path: "/v1/messages",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens: inputTokensCount,
      status: 502,
      success: false,
      error: "Upstream error",
    });
    return jsonResponse({ type: "error", error: { type: "api_error", message: "Upstream error" } }, 502);
  }

  const messageId = `msg_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;

  if (stream) {
    // Anthropic 流式响应格式
    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        const reader = tempoRes.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let inputTokens = internalMessages.reduce((acc, m) => acc + Math.ceil(m.content.length / 4), 0);

        // 发送 message_start
        const messageStart = {
          type: "message_start",
          message: {
            id: messageId,
            type: "message",
            role: "assistant",
            content: [],
            model: baseModel,
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: inputTokens, output_tokens: 0 },
          },
        };
        controller.enqueue(encoder.encode(`event: message_start\ndata: ${JSON.stringify(messageStart)}\n\n`));

        // 发送 content_block_start
        const blockStart = { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } };
        controller.enqueue(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`));

        let outputTokens = 0;

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;
              const delta = parseTempoChunk(line.trim());
              if (delta) {
                outputTokens += Math.ceil(delta.chunk.length / 4);
                const contentDelta = { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: delta.chunk } };
                controller.enqueue(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify(contentDelta)}\n\n`));
              }
            }
          }

          // 发送 content_block_stop
          const blockStop = { type: "content_block_stop", index: 0 };
          controller.enqueue(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`));

          // 发送 message_delta
          const messageDelta = {
            type: "message_delta",
            delta: { stop_reason: "end_turn", stop_sequence: null },
            usage: { output_tokens: outputTokens },
          };
          controller.enqueue(encoder.encode(`event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`));

          // 发送 message_stop
          controller.enqueue(encoder.encode(`event: message_stop\ndata: {"type":"message_stop"}\n\n`));
          
          // Record successful streaming request
          const duration = Date.now() - startTime;
          getStatsCollector().recordRequest(baseModel, duration, true);
          
          // Enhanced logging for streaming response
          logRequest({
            method: "POST",
            path: "/v1/messages",
            model: baseModel,
            reasoning,
            search,
            canvasId,
            duration,
            inputTokens,
            outputTokens,
            status: 200,
            success: true,
          });
        } finally {
          controller.close();
          reader.releaseLock();
        }
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } else {
    // 非流式响应
    const reader = tempoRes.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const delta = parseTempoChunk(line.trim());
        if (delta) fullText += delta.chunk;
      }
    }

    // Record successful non-streaming request
    const duration = Date.now() - startTime;
    getStatsCollector().recordRequest(baseModel, duration, true);

    const inputTokens = inputTokensCount;
    const outputTokens = estimateTokens(fullText);
    
    // Enhanced logging for non-streaming response
    logRequest({
      method: "POST",
      path: "/v1/messages",
      model: baseModel,
      reasoning,
      search,
      canvasId,
      duration,
      inputTokens,
      outputTokens,
      status: 200,
      success: true,
    });

    return jsonResponse({
      id: messageId,
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: fullText }],
      model: baseModel,
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: outputTokens },
    });
  }
}

// ============== Health Check ==============

// Version constant for health endpoint
const VERSION = "1.0.0";

// Service start time for uptime calculation
const SERVICE_START_TIME = Date.now();

/**
 * Health check response interface
 */
interface HealthResponse {
  status: "ok" | "degraded" | "error";
  uptime: number;
  version: string;
  tokenStatus: "valid" | "expired" | "unknown";
}

/**
 * Check if the current auth token is valid
 * @returns Token status: "valid", "expired", or "unknown"
 */
function checkTokenStatus(): "valid" | "expired" | "unknown" {
  if (!authToken) {
    return "unknown";
  }
  
  // Check if token is expired
  if (tokenExpiry > 0 && Date.now() >= tokenExpiry) {
    return "expired";
  }
  
  // Token exists and is not expired
  if (tokenExpiry > 0) {
    return "valid";
  }
  
  return "unknown";
}

/**
 * Handle health check request
 * Returns service status, uptime, version, and token validity
 * 
 * Requirements 7.1, 7.2, 7.3:
 * - Return service status
 * - Return 200 with status "ok" and uptime when healthy
 * - Return 200 with status "degraded" when auth token is invalid/expired
 */
function handleHealthCheck(): Response {
  const uptimeSeconds = Math.floor((Date.now() - SERVICE_START_TIME) / 1000);
  const tokenStatus = checkTokenStatus();
  
  // Determine overall status based on token validity
  let status: "ok" | "degraded" | "error";
  if (tokenStatus === "valid") {
    status = "ok";
  } else if (tokenStatus === "expired" || tokenStatus === "unknown") {
    status = "degraded";
  } else {
    status = "error";
  }
  
  const response: HealthResponse = {
    status,
    uptime: uptimeSeconds,
    version: VERSION,
    tokenStatus,
  };
  
  return jsonResponse(response, 200);
}

// ============== Auth Middleware ==============

/**
 * Check API key authentication for a request
 * Returns a 401 response if authentication fails, null otherwise
 * 
 * Requirements: 9.1, 9.2, 9.3
 * - Skip if PROXY_API_KEY not set
 * - Validate API key if set
 * - Return 401 for invalid key
 */
function checkAuth(req: Request): Response | null {
  const result = validateRequest(req);

  if (!result.valid) {
    console.log(`[Auth] Request rejected: ${result.error}`);
    return new Response(
      JSON.stringify({
        error: {
          message: result.error || "Invalid API key",
          type: "authentication_error",
        },
      }),
      {
        status: 401,
        headers: {
          "Content-Type": "application/json",
          "WWW-Authenticate": "Bearer",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }

  return null;
}

// ============== Rate Limit Middleware ==============

/**
 * Check rate limit for a request
 * Returns a 429 response if limit exceeded, null otherwise
 */
function checkRateLimit(req: Request): Response | null {
  const rateLimiter = getRateLimiter();
  
  // If rate limiting is disabled, allow all requests
  if (!rateLimiter.isEnabled()) {
    return null;
  }

  const clientIp = getClientIp(req);
  const result = rateLimiter.checkLimit(clientIp);

  if (!result.allowed) {
    console.log(`[RateLimit] Client ${clientIp} exceeded rate limit, retry after ${result.retryAfter}s`);
    return new Response(
      JSON.stringify({
        error: {
          message: "Rate limit exceeded",
          type: "rate_limit_error",
          retryAfter: result.retryAfter,
        },
      }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Retry-After": String(result.retryAfter),
          "X-RateLimit-Remaining": "0",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }

  // Record the request
  rateLimiter.recordRequest(clientIp);
  return null;
}

// ============== Queue Middleware ==============

/**
 * Create a 503 Service Unavailable response when queue is full
 */
function createQueueFullResponse(): Response {
  return new Response(
    JSON.stringify({
      error: {
        message: "Service busy",
        type: "service_unavailable",
      },
    }),
    {
      status: 503,
      headers: {
        "Content-Type": "application/json",
        "Retry-After": "5",
        "Access-Control-Allow-Origin": "*",
      },
    }
  );
}

/**
 * Wrap a request handler with queue management
 * Enqueues requests when at capacity, returns 503 when queue is full
 */
async function withQueue<T>(task: () => Promise<T>): Promise<T | Response> {
  const queue = getRequestQueue();
  
  try {
    return await queue.enqueue(task);
  } catch (error) {
    if (error instanceof QueueFullError) {
      console.log("[Queue] Request rejected - queue is full");
      return createQueueFullResponse();
    }
    throw error;
  }
}

// ============== 路由 ==============

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const path = url.pathname;

  // CORS 预检
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key, anthropic-version",
      },
    });
  }

  console.log(`[${new Date().toISOString()}] ${req.method} ${path}`);

  // Check API key authentication for API endpoints (not for OPTIONS, health, or stats)
  if (path.startsWith("/v1/")) {
    const authResponse = checkAuth(req);
    if (authResponse) {
      return authResponse;
    }
  }

  // Check rate limit for API endpoints (not for OPTIONS or static endpoints)
  if (path.startsWith("/v1/")) {
    const rateLimitResponse = checkRateLimit(req);
    if (rateLimitResponse) {
      return rateLimitResponse;
    }
  }

  try {
    // Health check endpoint - Requirements 7.1, 7.2, 7.3
    if (path === "/health" && req.method === "GET") {
      return handleHealthCheck();
    }

    // Stats endpoint - Requirements 8.1
    if (path === "/stats" && req.method === "GET") {
      const stats = getStatsCollector().getStats();
      return jsonResponse(stats);
    }

    // OpenAI 兼容接口
    if (path === "/v1/models" && req.method === "GET") {
      return jsonResponse(getModels());
    }
    if (path === "/v1/chat/completions" && req.method === "POST") {
      // Wrap with queue to manage concurrency
      const result = await withQueue(() => handleChat(req));
      if (result instanceof Response) {
        return result;
      }
      return result;
    }

    // Anthropic 兼容接口 (Claude Code)
    if (path === "/v1/messages" && req.method === "POST") {
      // Wrap with queue to manage concurrency
      const result = await withQueue(() => handleAnthropicMessages(req));
      if (result instanceof Response) {
        return result;
      }
      return result;
    }

    return jsonResponse({ error: { message: "Not found", type: "invalid_request_error" } }, 404);
  } catch (err) {
    console.error("[Error]", err);
    return jsonResponse({ error: { message: "Internal error", type: "api_error" } }, 500);
  }
}

// ============== 启动服务 ==============

console.log(`
╔═══════════════════════════════════════════════╗
║       Tempo API Proxy (Deno 版)               ║
╠═══════════════════════════════════════════════╣
║  端口: ${CONFIG.port}                                ║
║                                               ║
║  OpenAI 兼容:                                 ║
║    POST /v1/chat/completions                  ║
║    GET  /v1/models                            ║
║                                               ║
║  Anthropic 兼容 (Claude Code):                ║
║    POST /v1/messages                          ║
║                                               ║
║  管理端点:                                    ║
║    GET  /health                               ║
║    GET  /stats                                ║
╚═══════════════════════════════════════════════╝
`);

Deno.serve({ port: CONFIG.port }, handler);

