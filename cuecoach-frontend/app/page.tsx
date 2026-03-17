"use client";

import { useState } from "react";

import ChatInput from "@/components/chat-input";
import ChatWindow from "@/components/chat-window";
import ModeToggle from "@/components/mode-toggle";
import { askCueCoach } from "@/lib/api";
import type { ChatMessage, ChatMode } from "@/types/chat";

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<ChatMode>("explain");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");

  async function handleSend(question: string) {
    const userMessage: ChatMessage = {
      role: "user",
      content: question,
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setError("");
    setIsLoading(true);

    try {
      const response = await askCueCoach({
        question,
        mode,
        top_k: 5,
        min_score: 0.42,
        max_context_chars: 12000,
        chat_history: messages,
      });

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.answer,
      };

      setMessages([...nextMessages, assistantMessage]);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Something went wrong.";

      setError(message);

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: "I ran into an error talking to the backend.",
      };

      setMessages([...nextMessages, assistantMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleClearChat() {
    setMessages([]);
    setError("");
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-sm md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-slate-100">
              CueCoach
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Ask questions related to billiards
            </p>
          </div>

          <div className="flex flex-col items-start gap-3 md:items-end">
            <ModeToggle value={mode} onChange={setMode} />
            <button
              type="button"
              onClick={handleClearChat}
              className="text-sm font-medium text-slate-400 hover:text-slate-100"
            >
              Clear chat
            </button>
          </div>
        </div>

        <ChatWindow messages={messages} isLoading={isLoading} />

        <ChatInput onSend={handleSend} disabled={isLoading} />

        {error ? (
          <div className="mt-4 rounded-2xl border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        ) : null}
      </div>
    </main>
  );
}