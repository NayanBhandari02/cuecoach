export type ChatMode = "strict" | "explain";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type AskRequest = {
  question: string;
  mode: ChatMode;
  top_k: number;
  min_score: number;
  max_context_chars: number;
  chat_history: ChatMessage[];
};

export type AskResponse = {
  answer: string;
  mode: ChatMode;
};